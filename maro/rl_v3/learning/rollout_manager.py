# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from collections import defaultdict
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from os import getpid
from random import choices
from typing import Callable, Dict, List, Tuple

import numpy as np

from maro.communication import Proxy, SessionType
from maro.rl.utils import MsgKey, MsgTag
from maro.utils import DummyLogger, Logger, set_seeds

from .env_sampler import AbsEnvSampler
from .helpers import get_rollout_finish_msg


def concat_batches(batch_list: List[dict]) -> dict:
    return {key: np.concatenate([batch[key] for batch in batch_list]) for key in batch_list[0]}


class AbsRolloutManager(object):
    """Controller for simulation data collection."""
    def __init__(self) -> None:
        super(AbsRolloutManager, self).__init__()
        self._end_of_episode = False

    @abstractmethod
    def collect(self, ep: int, segment: int, policy_state_dict: dict) -> Tuple[Dict, List[Dict]]:
        """Collect simulation data, i.e., experiences for training.

        Args:
            ep (int): Current episode.
            segment (int): Current segment.
            policy_state_dict (dict): Policy states to use for collecting training info.

        Returns:
            A 2-tuple consisting of a dictionary of roll-out information grouped by policy ID and a list of dictionaries
            containing step-level information collected by the user-defined ``post_step`` callback in ``AbsEnvSampler``.
            An RL policy's roll-out information must be either loss information or a data batch that can be passed to
            the policy's ``update`` or ``learn``, respectively.
        """
        raise NotImplementedError

    @property
    def end_of_episode(self) -> bool:
        return self._end_of_episode

    @abstractmethod
    def evaluate(self, ep: int, policy_state_dict: dict) -> list:
        """Evaluate policy performance.

        Args:
            ep (int): Current training episode.
            policy_state_dict (dict): Policy states to use for evaluation.

        Returns:
            A list of dictionaries containing step-level information collected by the user-defined ``post_step``
            callback in ``AbsEnvSampler`` for evaluation purposes.
        """
        raise NotImplementedError

    def reset(self) -> None:
        self._end_of_episode = False

    @abstractmethod
    def exit(self) -> None:
        raise NotImplementedError


class MultiProcessRolloutManager(AbsRolloutManager):
    """Local roll-out controller.

    Args:
        get_env_sampler (Callable): Function to create an environment sampler for collecting training data. The function
            should take no parameters and return an ``AbsEnvSampler`` instance.
        num_rollouts (int): Number of processes to spawn for parallel roll-out.
        num_steps (int): Number of environment steps to roll out in each call to ``collect``. Defaults to -1, in which
            case the roll-out will be executed until the end of the environment.
        num_eval_rollouts (int): Number of roll-out processes to use for evaluation. Defaults to 1.
        logger (Logger): A ``Logger`` instance for logging important events. Defaults to a ``DummyLogger``
            which logs nothing.
    """
    def __init__(
        self,
        get_env_sampler: Callable[[], AbsEnvSampler],
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

        def _rollout_worker(_index: int, conn: Connection, _get_env_sampler: Callable) -> None:
            set_seeds(_index)
            env_sampler = _get_env_sampler()
            self._logger.info(f"Roll-out worker {_index} started with PID {getpid()}")
            while True:
                msg = conn.recv()
                if msg["type"] == "sample":
                    result = env_sampler.sample(policy_state_dict=msg["policy_state"], num_steps=self._num_steps)
                    self._logger.info(get_rollout_finish_msg(
                        msg["episode"], result["step_range"], exploration_params=result["exploration_params"]
                    ))
                    result["worker_index"] = _index
                    conn.send(result)
                elif msg["type"] == "test":
                    self._logger.info("Evaluating...")
                    tracker = env_sampler.test(msg["policy_state"])
                    conn.send({"worker_id": _index, "tracker": tracker})
                elif msg["type"] == "quit":
                    break

        for index in range(self._num_rollouts):
            manager_end, worker_end = Pipe()
            self._manager_ends.append(manager_end)
            worker = Process(target=_rollout_worker, args=(index, worker_end, get_env_sampler), daemon=True)
            self._worker_processes.append(worker)
            worker.start()

    def collect(self, ep: int, segment: int, policy_state_dict: dict) -> Tuple[Dict, List[Dict]]:
        self._logger.info(f"Collecting simulation data (episode {ep}, segment {segment})")

        info_by_policy, trackers = defaultdict(list), []
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
            for policy_id, info in result["rollout_info"].items():
                info_by_policy[policy_id].append(info)
            trackers.append(result["tracker"])
            self._end_of_episode = result["end_of_episode"]

        # concat batches from different roll-out workers
        new_info_by_policy = {k: v for k, v in info_by_policy.items()}
        for policy_id, info_list in new_info_by_policy.items():
            if "loss" not in info_list[0]:
                new_info_by_policy[policy_id] = concat_batches(info_list)

        return new_info_by_policy, trackers

    def evaluate(self, ep: int, policy_state_dict: dict) -> list:
        trackers = []
        eval_worker_conns = choices(self._manager_ends, k=self._num_eval_rollouts)
        for conn in eval_worker_conns:
            conn.send({"type": "test", "policy_state": policy_state_dict})
        for conn in eval_worker_conns:
            result = conn.recv()
            trackers.append(result["tracker"])

        return trackers

    def exit(self) -> None:
        """Tell the worker processes to exit."""
        for conn in self._manager_ends:
            conn.send({"type": "quit"})


class DistributedRolloutManager(AbsRolloutManager):
    """Controller for a set of remote roll-out workers, possibly distributed on different computation nodes.

    Args:
        group (str): Group name for the roll-out cluster, which includes all roll-out workers and a roll-out manager
            that manages them.
        num_workers (int): Number of remote roll-out workers.
        num_steps (int): Number of environment steps to roll out in each call to ``collect``. Defaults to -1, in which
            case the roll-out will be executed until the end of the environment.
        min_finished_workers (int): Minimum number of finished workers required for a ``collect`` call. Defaults to
            None, in which case it will be set to ``num_workers``.
        max_extra_recv_tries (int): Maximum number of attempts to receive worker results after ``min_finished_workers``
            have been received in ``collect``. Defaults to 0.
        extra_recv_timeout (int): Timeout (in milliseconds) for each attempt to receive from a worker after
            ``min_finished_workers`` have been received in ``collect``. Defaults to 100 (milliseconds).
        num_eval_workers (int): Number of workers for evaluation. Defaults to 1.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to the empty dictionary.
        logger (Logger): A ``Logger`` instance for logging important events. Defaults to a ``DummyLogger``
            which logs nothing.
    """
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

    def collect(self, ep: int, segment: int, policy_state_dict: dict) -> Tuple[Dict, List[Dict]]:
        msg_body = {
            MsgKey.EPISODE: ep,
            MsgKey.SEGMENT: segment,
            MsgKey.NUM_STEPS: self._num_steps,
            MsgKey.POLICY_STATE: policy_state_dict
        }

        self._proxy.iscatter(MsgTag.SAMPLE, SessionType.TASK, [(worker_id, msg_body) for worker_id in self._workers])
        self._logger.info(f"Collecting simulation data (episode {ep}, segment {segment})")

        info_by_policy, trackers, num_finishes = defaultdict(list), [], 0
        # Ensure the minimum number of worker results are received.
        for msg in self._proxy.receive():
            rollout_info, tracker = self._handle_worker_result(msg, ep, segment)
            if rollout_info:
                num_finishes += 1
                for policy_id, info in rollout_info.items():
                    info_by_policy[policy_id].append(info)
                trackers.append(tracker)
            if num_finishes == self._min_finished_workers:
                break

        # Keep trying to receive from workers, but with timeout
        for i in range(self._max_extra_recv_tries):
            msg = self._proxy.receive_once(timeout=self._extra_recv_timeout)
            if not msg:
                self._logger.info(f"Receive timeout, {self._max_extra_recv_tries - i - 1} attempts left")
            else:
                rollout_info, tracker = self._handle_worker_result(msg, ep, segment)
                if rollout_info:
                    num_finishes += 1
                    for policy_id, info in rollout_info.items():
                        info_by_policy[policy_id].append(info)
                    trackers.append(tracker)
                if num_finishes == self._num_workers:
                    break

        # concat batches from different roll-out workers
        new_info_by_policy = {k: v for k, v in info_by_policy.items()}
        for policy_id, info_list in new_info_by_policy.items():
            if "loss" not in info_list[0]:
                new_info_by_policy[policy_id] = concat_batches(info_list)

        return new_info_by_policy, trackers

    def _handle_worker_result(self, msg, ep, segment) -> tuple:
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

    def evaluate(self, ep: int, policy_state_dict: dict) -> list:
        msg_body = {MsgKey.EPISODE: ep, MsgKey.POLICY_STATE: policy_state_dict}

        workers = choices(self._workers, k=self._num_eval_workers)
        self._proxy.iscatter(MsgTag.TEST, SessionType.TASK, [(worker_id, msg_body) for worker_id in workers])
        self._logger.info(f"Sent evaluation requests to {workers}")

        # Receive roll-out results from remote workers
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
        """Tell the remote workers to exit."""
        self._proxy.ibroadcast("rollout_worker", MsgTag.EXIT, SessionType.NOTIFICATION)
        self._proxy.close()
        self._logger.info("Exiting...")
