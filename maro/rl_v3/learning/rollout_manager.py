# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from collections import defaultdict
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from os import getpid
from random import choices
from typing import Callable, Dict, List, Optional, Tuple

from maro.communication import Proxy, SessionType
from maro.rl.utils import MsgKey, MsgTag
from maro.utils import DummyLogger, Logger, set_seeds

from .env_sampler import AbsEnvSampler, ExpElement


class AbsRolloutManager(object):
    def __init__(self) -> None:
        super(AbsRolloutManager, self).__init__()
        self._end_of_episode = False

    @property
    def end_of_episode(self) -> bool:
        return self._end_of_episode

    @abstractmethod
    def collect(
        self, ep: int, segment: int, policy_state_dict: Dict[str, object]
    ) -> Tuple[List[List[ExpElement]], List[dict]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, ep: int, policy_state_dict: Dict[str, object]) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    def exit(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        self._end_of_episode = False


class MultiProcessRolloutManager(AbsRolloutManager):
    def __init__(
        self,
        get_env_sampler_func: Callable[[], AbsEnvSampler],
        num_rollouts: int,
        num_steps: int = -1,
        num_eval_rollouts: int = 1,
        logger: Logger = DummyLogger()
    ) -> None:
        super(MultiProcessRolloutManager, self).__init__()

        if num_steps == 0 or num_steps < -1:
            raise ValueError("num_steps must be a positive integer or -1.")
        if num_rollouts <= 1:
            raise ValueError("'num_rollouts' must be greater than 1.")
        if num_eval_rollouts > num_rollouts:
            raise ValueError("'num_eval_rollouts' can not be greater than 'num_rollouts'.")

        self._logger = logger
        self._num_steps = num_steps if num_steps > 0 else float("inf")
        self._num_rollouts = num_rollouts
        self._num_eval_rollouts = num_eval_rollouts
        self._worker_processes = []
        self._manager_ends = []

        def _rollout_worker(_index: int, _conn: Connection, _get_env_sampler_func: Callable[[], AbsEnvSampler]) -> None:
            set_seeds(_index)
            env_sampler = _get_env_sampler_func()
            self._logger.info(f"Roll-out worker {_index} started with PID {getpid()}")

            while True:
                msg = _conn.recv()
                if msg["type"] == "sample":
                    self._logger.info("Sampling...")
                    result = env_sampler.sample(policy_state_dict=msg["policy_state"], num_steps=self._num_steps)
                    # self._logger.info(get_rollout_finish_msg(
                    #     msg["episode"], result["step_range"], exploration_params=result["exploration_params"]
                    # ))
                    result["worker_index"] = index
                    _conn.send(result)
                elif msg["type"] == "test":
                    self._logger.info("Evaluating...")
                    tracker = env_sampler.test(msg["policy_state"])
                    _conn.send({"worker_id": index, "tracker": tracker})
                elif msg["type"] == "quit":
                    break

        for index in range(self._num_rollouts):
            print('Huoran', index)
            manager_end, worker_end = Pipe()
            self._manager_ends.append(manager_end)
            worker = Process(target=_rollout_worker, args=(index, worker_end, get_env_sampler_func), daemon=True)
            self._worker_processes.append(worker)
            worker.start()

    def collect(
        self, ep: int, segment: int, policy_state_dict: Dict[str, object]
    ) -> Tuple[List[List[ExpElement]], List[dict]]:
        self._logger.info(f"Collecting simulation data (episode {ep}, segment {segment})")

        exp_lists, trackers = [], []
        rollout_req = {
            "type": "sample",
            "episode": ep,
            "num_steps": self._num_steps,
            "policy_state": policy_state_dict
        }

        for conn in self._manager_ends:
            conn.send(rollout_req)
        for conn in self._manager_ends:
            result = conn.recv()
            exp_lists.append(result["experiences"])
            self._end_of_episode = result["end_of_episode"]  # TODO: overwrite?
            trackers.append(result["tracker"])

        return exp_lists, trackers

    def evaluate(self, ep: int, policy_state_dict: Dict[str, object]) -> List[dict]:
        trackers = []
        eval_worker_conns = choices(self._manager_ends, k=self._num_eval_rollouts)

        rollout_req = {"type": "test", "policy_state": policy_state_dict}

        for conn in eval_worker_conns:
            conn.send(rollout_req)
        for conn in eval_worker_conns:
            result = conn.recv()
            trackers.append(result["tracker"])

        return trackers

    def exit(self) -> None:
        rollout_req = {"type": "quit"}
        for conn in self._manager_ends:
            conn.send(rollout_req)


class DistributedRolloutManager(AbsRolloutManager):
    def __init__(
        self,
        group: str,
        num_workers: int,
        num_steps: int = -1,
        min_finished_workers: int = None,
        max_extra_recv_tries: int = 0,
        extra_recv_timeout: int = 100,
        max_lag: Dict[str, int] = None,
        num_eval_workers: int = 1,
        proxy_kwargs: dict = None,
        logger: Logger = DummyLogger()
    ) -> None:
        super(DistributedRolloutManager, self).__init__()

        if max_lag is None:
            max_lag = defaultdict(int)
        if proxy_kwargs is None:
            proxy_kwargs = {}

        if num_eval_workers > num_workers:
            raise ValueError("num_eval_workers cannot exceed the number of available workers")

        self._num_workers = num_workers
        peers = {"rollout_worker": num_workers}
        self._proxy = Proxy(group, "rollout_manager", peers, component_name="ROLLOUT_MANAGER", **proxy_kwargs)
        self._workers = self._proxy.peers["rollout_worker"]  # remote roll-out worker ID's
        self._logger = logger

        self._num_steps = num_steps
        if min_finished_workers is None:
            min_finished_workers = self._num_workers
            self._logger.info(f"Minimum number of finished workers is set to {min_finished_workers}")

        self._min_finished_workers = min_finished_workers

        if max_extra_recv_tries is None:
            max_extra_recv_tries = self._num_workers - self._min_finished_workers
            self._logger.info(f"Maximum number of extra receive tries is set to {max_extra_recv_tries}")

        self._max_extra_recv_tries = max_extra_recv_tries
        self._extra_recv_timeout = extra_recv_timeout

        self._max_lag = max_lag
        self._num_eval_workers = num_eval_workers

    def collect(
        self, ep: int, segment: int, policy_state_dict: Dict[str, object]
    ) -> Tuple[List[List[ExpElement]], List[dict]]:
        msg_body = {
            MsgKey.EPISODE: ep,
            MsgKey.SEGMENT: segment,
            MsgKey.NUM_STEPS: self._num_steps,
            MsgKey.POLICY_STATE: policy_state_dict
        }

        self._proxy.iscatter(MsgTag.SAMPLE, SessionType.TASK, [(worker_id, msg_body) for worker_id in self._workers])
        self._logger.info(f"Collecting simulation data (episode {ep}, segment {segment})")

        exp_lists, trackers, num_finishes = [], [], 0
        for msg in self._proxy.receive():
            exp_list, tracker = self._handle_worker_result(msg, ep, segment)
            if exp_list is not None:
                num_finishes += 1
                exp_lists.append(exp_list)
                trackers.append(tracker)
                if num_finishes == self._min_finished_workers:
                    break

        for i in range(self._max_extra_recv_tries):
            msg = self._proxy.receive_once(timeout=self._extra_recv_timeout)
            if not msg:
                self._logger.info(f"Receive timeout, {self._max_extra_recv_tries - i - 1} attempts left")
            else:
                exp_list, tracker = self._handle_worker_result(msg, ep, segment)
                if exp_list is not None:
                    num_finishes += 1
                    exp_lists.append(exp_list)
                    trackers.append(tracker)
                    if num_finishes == self._min_finished_workers:
                        break

        return exp_lists, trackers

    def _handle_worker_result(
        self, msg, ep: int, segment: int
    ) -> Tuple[Optional[List[ExpElement]], Optional[dict]]:  # TODO: msg type
        if msg.tag != MsgTag.SAMPLE_DONE:
            self._logger.info(
                f"Ignored a message of type {msg.tag} (expected message type {MsgTag.SAMPLE_DONE})"
            )
            return None, None

        # The message is what we expect
        if msg.body[MsgKey.EPISODE] == ep and msg.body[MsgKey.SEGMENT] == segment:
            self._end_of_episode = msg.body[MsgKey.END_OF_EPISODE]
            return msg.body[MsgKey.ROLLOUT_INFO], msg.body[MsgKey.TRACKER]

        return None, None

    def evaluate(self, ep: int, policy_state_dict: Dict[str, object]) -> List[dict]:
        msg_body = {
            MsgKey.EPISODE: ep,
            MsgKey.POLICY_STATE: policy_state_dict
        }
        workers = choices(self._workers, k=self._num_eval_workers)
        self._proxy.iscatter(MsgTag.TEST, SessionType.TASK, [(worker_id, msg_body) for worker_id in workers])
        self._logger.info(f"Sent evaluation requests to {workers}")

        num_finishes = 0
        trackers = []
        for msg in self._proxy.receive():
            if msg.tag != MsgTag.TEST_DONE or msg.body[MsgKey.EPISODE] != ep:
                self._logger.info(
                    f"Ignore a message of type {msg.tag} with episode {msg.body[MsgKey.EPISODE]} "
                    f"(expected message type {MsgTag.TEST_DONE} and episode {ep})"
                )
                continue

            trackers.append(msg.body[MsgKey.TRACKER])
            if msg.body[MsgKey.EPISODE] == ep:
                num_finishes += 1
                if num_finishes == self._num_eval_workers:
                    break

        return trackers

    def exit(self) -> None:
        self._proxy.ibroadcast("rollout_worker", MsgTag.EXIT, SessionType.NOTIFICATION)
        self._proxy.close()
        self._logger.info("Exiting...")
