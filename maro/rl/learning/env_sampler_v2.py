# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from multiprocessing import Pipe, Process
from os import getcwd, path
from typing import Callable, Dict

import numpy as np

from maro.communication import Proxy, SessionMessage, SessionType
from maro.rl.policy_v2 import RLPolicy
from maro.rl.utils import MsgKey, MsgTag
from maro.simulator import Env
from maro.utils import Logger, clone

from .helpers import get_rollout_finish_msg


class AbsAgentWrapper(object):
    def __init__(self, aid2pid: Dict[str, str]) -> None:
        self._aid2pid = aid2pid

    @abstractmethod
    def load(self, dir: str) -> None:
        pass

    @abstractmethod
    def choose_action(self, states_by_agent_id: Dict[str, np.ndarray]) -> dict:
        pass

    @abstractmethod
    def set_policy_states(self, states_by_policy_id: dict) -> None:
        pass

    @abstractmethod
    def explore(self) -> None:
        pass

    @abstractmethod
    def exploit(self) -> None:
        pass

    @abstractmethod
    def exploration_step(self) -> None:
        pass

    @abstractmethod
    def get_rollout_info(self) -> dict:
        pass

    @abstractmethod
    def get_exploration_params(self) -> dict:
        pass

    @abstractmethod
    def record_transition(
        self, agent: str, state: np.ndarray, action: dict, reward: float, next_state: np.ndarray, terminal: bool
    ) -> None:
        pass

    @abstractmethod
    def improve(self, checkpoint_dir: str = None) -> None:
        pass


class SimpleAgentWrapper(AbsAgentWrapper):
    """Wrapper for multiple agents using multiple policies to expose simple single-agent interfaces."""
    def __init__(self, get_policy_func_dict: Dict[str, Callable], aid2pid: Dict[str, str]) -> None:
        super(SimpleAgentWrapper, self).__init__(aid2pid)

        self._pid2policy: Dict[str, RLPolicy] = {
            policy_id: func(policy_id) for policy_id, func in get_policy_func_dict.items()
        }
        self._aid2policy: Dict[str, RLPolicy] = {
            agent_id: self._pid2policy[policy_id] for agent_id, policy_id in self._aid2pid.items()
        }

    def load(self, dir: str) -> None:
        for policy_id, policy in self._pid2policy.items():
            pth = path.join(dir, policy_id)
            if path.exists(pth):
                policy.load(pth)

    def choose_action(self, states_by_agent_id: Dict[str, np.ndarray]) -> dict:
        pid2states, pid2aids = defaultdict(list), defaultdict(list)
        for agent_id, state in states_by_agent_id.items():
            policy_id = self._aid2pid[agent_id]
            pid2states[policy_id].append(state)
            pid2aids[policy_id].append(agent_id)

        actions_by_agent_id = {}
        # compute the actions for local policies first while the inferences processes do their work.
        for policy_id, policy in self._pid2policy.items():
            if pid2states[policy_id]:
                actions_by_agent_id.update(
                    zip(pid2aids[policy_id], policy(np.vstack(pid2states[policy_id])))
                )

        return actions_by_agent_id

    def set_policy_states(self, states_by_policy_id: dict) -> None:
        for policy_id, state in states_by_policy_id.items():
            self._pid2policy[policy_id].set_state(state)

    def explore(self) -> None:
        for policy in self._pid2policy.values():
            policy.explore()

    def exploit(self) -> None:
        for policy in self._pid2policy.values():
            policy.exploit()

    def exploration_step(self) -> None:
        for policy in self._pid2policy.values():
            if hasattr(policy, "exploration_step"):
                policy.exploration_step()

    def get_rollout_info(self) -> dict:
        return {
            policy_id: policy.get_rollout_info() for policy_id, policy in self._pid2policy.items()
            if isinstance(policy, RLPolicy)
        }

    def get_exploration_params(self) -> dict:
        return {
            policy_id: clone(policy.exploration_params) for policy_id, policy in self._pid2policy.items()
            if isinstance(policy, RLPolicy)
        }

    def record_transition(
        self, agent: str, state: np.ndarray, action: dict, reward: float, next_state: np.ndarray, terminal: bool
    ) -> None:
        if isinstance(self._aid2policy[agent], RLPolicy):
            self._aid2policy[agent].record(agent, state, action, reward, next_state, terminal)

    def improve(self, checkpoint_dir: str = None) -> None:
        for policy_id, policy in self._pid2policy.items():
            if hasattr(policy, "improve"):
                policy.improve()
                if checkpoint_dir:
                    policy.save(path.join(checkpoint_dir, policy_id))


class ParallelAgentWrapper(AbsAgentWrapper):
    """Wrapper for multiple agents using multiple policies to expose simple single-agent interfaces.

    The policy instances are distributed across multiple processes to achieve parallel inference.
    """
    def __init__(self, get_policy_func_dict: Dict[str, Callable], aid2pid: Dict[str, str]) -> None:
        super(ParallelAgentWrapper, self).__init__(aid2pid)

        self._inference_services = []
        self._conn = {}

        def _inference_service(id_, get_policy, conn) -> None:
            policy = get_policy(id_)
            while True:
                msg = conn.recv()
                if msg["type"] == "load":
                    if hasattr(policy, "load"):
                        policy.load(path.join(msg["dir"], id_))
                elif msg["type"] == "choose_action":
                    actions = policy(msg["states"])
                    conn.send(actions)
                elif msg["type"] == "set_state":
                    if hasattr(policy, "set_state"):
                        policy.set_state(msg["policy_state"])
                elif msg["type"] == "explore":
                    policy.explore()
                elif msg["type"] == "exploit":
                    policy.exploit()
                elif msg["type"] == "exploration_step":
                    if hasattr(policy, "exploration_step"):
                        policy.exploration_step()
                elif msg["type"] == "rollout_info":
                    conn.send(policy.get_rollout_info() if hasattr(policy, "get_rollout_info") else None)
                elif msg["type"] == "exploration_params":
                    conn.send(policy.exploration_params if hasattr(policy, "exploration_params") else None)
                elif msg["type"] == "record":
                    if hasattr(policy, "record"):
                        policy.record(
                            msg["agent"], msg["state"], msg["action"], msg["reward"], msg["next_state"], msg["terminal"]
                        )
                elif msg["type"] == "update":
                    if hasattr(policy, "update"):
                        policy.update(msg["loss_info"])
                elif msg["type"] == "learn":
                    if hasattr(policy, "learn"):
                        policy.learn(msg["batch"])
                elif msg["type"] == "improve":
                    if hasattr(policy, "improve"):
                        policy.improve()
                        if msg["checkpoint_dir"]:
                            policy.save(path.join(msg["checkpoint_dir"], id_))

        for policy_id in get_policy_func_dict:
            conn1, conn2 = Pipe()
            self._conn[policy_id] = conn1
            host = Process(
                target=_inference_service,
                args=(policy_id, get_policy_func_dict[policy_id], conn2)
            )
            self._inference_services.append(host)
            host.start()

    def load(self, dir: str) -> None:
        for conn in self._conn.values():
            conn.send({"type": "load", "dir": dir})

    def choose_action(self, states_by_agent_id: Dict[str, np.ndarray]) -> dict:
        pid2states, pid2aids = defaultdict(list), defaultdict(list)
        for agent_id, state in states_by_agent_id.items():
            policy_id = self._aid2pid[agent_id]
            pid2states[policy_id].append(state)
            pid2aids[policy_id].append(agent_id)

        # send state batch to inference processes for parallelized inference.
        for policy_id, conn in self._conn.items():
            if pid2states[policy_id]:
                conn.send({"type": "choose_action", "states": np.vstack(pid2states[policy_id])})

        action_by_agent = {}
        for policy_id, conn in self._conn.items():
            if pid2states[policy_id]:
                action_by_agent.update(zip(pid2aids[policy_id], conn.recv()))

        return action_by_agent

    def set_policy_states(self, states_by_policy_id: dict) -> None:
        for policy_id, conn in self._conn.items():
            conn.send({"type": "set_state", "policy_state": states_by_policy_id[policy_id]})

    def explore(self) -> None:
        for conn in self._conn.values():
            conn.send({"type": "explore"})

    def exploit(self) -> None:
        for conn in self._conn.values():
            conn.send({"type": "exploit"})

    def exploration_step(self) -> None:
        for conn in self._conn.values():
            conn.send({"type": "exploration_step"})

    def get_rollout_info(self) -> dict:
        rollout_info = {}
        for conn in self._conn.values():
            conn.send({"type": "rollout_info"})

        for policy_id, conn in self._conn.items():
            info = conn.recv()
            if info:
                rollout_info[policy_id] = info

        return rollout_info

    def get_exploration_params(self) -> dict:
        exploration_params = {}
        for conn in self._conn.values():
            conn.send({"type": "exploration_params"})

        for policy_id, conn in self._conn.items():
            params = conn.recv()
            if params:
                exploration_params[policy_id] = params

        return exploration_params

    def record_transition(
        self, agent: str, state: np.ndarray, action: dict, reward: float, next_state: np.ndarray, terminal: bool
    ) -> None:
        self._conn[self._aid2pid[agent]].send({
            "type": "record", "agent": agent, "state": state, "action": action, "reward": reward,
            "next_state": next_state, "terminal": terminal
        })

    def improve(self, checkpoint_dir: str = None) -> None:
        for conn in self._conn.values():
            conn.send({"type": "improve", "checkpoint_dir": checkpoint_dir})


class AbsEnvSampler(ABC):
    """Simulation data collector and policy evaluator.

    Args:
        get_env (Callable[[], Env]): Function to create an ``Env`` instance for collecting training data. The function
            should take no parameters and return an environment wrapper instance.
        get_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have policy name as the only parameter and return an ``AbsPolicy`` instance.
        agent2policy (Dict[str, str]): A dictionary that maps agent IDs to policy IDs, i.e., specifies the policy used
            by each agent.
        get_test_env (Callable): Function to create an ``Env`` instance for testing policy performance. The function
            should take no parameters and return an environment wrapper instance. If this is None, the training
            environment wrapper will be used for evaluation in the worker processes. Defaults to None.
        reward_eval_delay (int): Number of ticks required after a decision event to evaluate the reward
            for the action taken for that event. Defaults to 0, which means rewards are evaluated immediately
            after executing an action.
        parallel_inference (bool): If True, the policies will be placed in separate processes so that inference can be
            performed in parallel to speed up simulation. This is useful if some policies are big and take a long time
            to generate actions. Defaults to False.
    """
    def __init__(
        self,
        get_env: Callable[[], Env],
        get_policy_func_dict: Dict[str, Callable],
        agent2policy: Dict[str, str],
        get_test_env: Callable[[], Env] = None,
        reward_eval_delay: int = 0,
        parallel_inference: bool = False
    ) -> None:
        self._learn_env = get_env()
        self._test_env = get_test_env() if get_test_env else self._learn_env
        self._env = None

        agent_wrapper_cls = ParallelAgentWrapper if parallel_inference else SimpleAgentWrapper
        self._agent_wrapper: AbsAgentWrapper = agent_wrapper_cls(get_policy_func_dict, agent2policy)

        self._reward_eval_delay = reward_eval_delay
        self._state = None
        self._event = None
        self._step_index = 0

        self._transition_cache = defaultdict(deque)  # for caching transitions whose rewards have yet to be evaluated
        self._tracker = {}  # User-defined tracking information is placed here.

    @property
    def event(self) -> object:
        return self._event

    @property
    def agent_wrapper(self) -> AbsAgentWrapper:
        return self._agent_wrapper

    @abstractmethod
    def get_state(self, tick: int = None) -> dict:
        """Compute the state for a given tick.

        Args:
            tick (int): The tick for which to compute the environmental state. If computing the current state,
                use tick=self.env.tick.
        Returns:
            A dictionary with (agent ID, state) as key-value pairs.
        """
        raise NotImplementedError

    @abstractmethod
    def get_env_actions(self, action) -> dict:
        """Convert policy outputs to an action that can be executed by ``self.env.step()``."""
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, actions: list, tick: int) -> dict:
        """Evaluate the reward for an action.
        Args:
            actions (list): Actions.
            tick (int): Evaluate the reward for the actions that occured at the given tick. Each action in
                ``actions`` must be an Action object defined for the environment in question.

        Returns:
            A dictionary with (agent ID, reward) as key-value pairs.
        """
        raise NotImplementedError

    def sample(self, policy_state_dict: dict = None, num_steps: int = -1, return_rollout_info: bool = True) -> dict:
        self._env = self._learn_env
        if not self._state:
            # reset and get initial state
            self._env.reset()
            self._step_index = 0
            self._transition_cache.clear()
            self._tracker.clear()
            _, self._event, _ = self._env.step(None)
            self._state = self.get_state()

        # set policy states
        if policy_state_dict:
            self._agent_wrapper.set_policy_states(policy_state_dict)
        self._agent_wrapper.explore()

        starting_step_index = self._step_index + 1
        steps_to_go = float("inf") if num_steps == -1 else num_steps
        while self._state and steps_to_go > 0:
            action = self._agent_wrapper.choose_action(self._state)
            env_actions = self.get_env_actions(action)
            for agent, state in self._state.items():
                self._transition_cache[agent].append((state, action[agent], env_actions, self._env.tick))
            _, self._event, done = self._env.step(env_actions)
            self._state = None if done else self.get_state()
            self._step_index += 1
            steps_to_go -= 1

        """
        If this is the final step, evaluate rewards for all remaining events except the last.
        Otherwise, evaluate rewards only for events at least self.reward_eval_delay ticks ago.
        """
        for agent, cache in self._transition_cache.items():
            while cache and (not self._state or self._env.tick - cache[0][-1] >= self._reward_eval_delay):
                state, action, env_actions, tick = cache.popleft()
                reward = self.get_reward(env_actions, tick)
                self.post_step(state, action, env_actions, reward, tick)
                self._agent_wrapper.record_transition(
                    agent, state, action, reward[agent], cache[0][0] if cache else self._state,
                    not cache and not self._state
                )

        result = {
            "step_range": (starting_step_index, self._step_index),
            "tracker": self._tracker,
            "end_of_episode": not self._state,
            "exploration_params": self._agent_wrapper.get_exploration_params()
        }
        if return_rollout_info:
            result["rollout_info"] = self._agent_wrapper.get_rollout_info()

        if not self._state:
            self._agent_wrapper.exploration_step()
        return result

    def test(self, policy_state_dict: dict = None) -> dict:
        self._env = self._test_env
        # set policy states
        if policy_state_dict:
            self._agent_wrapper.set_policy_states(policy_state_dict)

        # Set policies to exploitation mode
        self._agent_wrapper.exploit()

        self._env.reset()
        terminal = False
        # get initial state
        _, self._event, _ = self._env.step(None)
        state = self.get_state()
        while not terminal:
            action = self._agent_wrapper.choose_action(state)
            env_actions = self.get_env_actions(action)
            _, self._event, terminal = self._env.step(env_actions)
            if not terminal:
                state = self.get_state()

        return self._tracker

    @abstractmethod
    def post_step(self, state: np.ndarray, action, env_actions, reward, tick):  # TODO: argu type
        """
        Gather any information you wish to track during a roll-out episode and store it in the ``tracker`` attribute.
        """
        pass

    def worker(
        self,
        group: str,
        index: int,
        num_extra_recv_attempts: int = 0,
        recv_timeout: int = 100,
        proxy_kwargs: dict = None,
        log_dir: str = getcwd()
    ):
        """Roll-out worker process that can be launched on separate computation nodes.

        Args:
            group (str): Group name for the roll-out cluster, which includes all roll-out workers and a roll-out manager
                that manages them.
            index (int): Worker index. The worker's ID in the cluster will be "ROLLOUT_WORKER.{worker_idx}".
                This is used for bookkeeping by the roll-out manager.
            num_extra_recv_attempts (int): Number of extra receive attempts after each received ``SAMPLE`` message. This
                is used to catch the worker up to the latest episode in case it trails the main learning loop by at
                least one full episode. Defaults to 0.
            recv_timeout (int): Timeout for the extra receive attempts. Defaults to 100 (miliseconds).
            proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
                for details. Defaults to the empty dictionary.
            log_dir (str): Directory to store logs in. Defaults to the current working directory.
        """
        if proxy_kwargs is None:
            proxy_kwargs = {}

        proxy = Proxy(
            group, "rollout_worker", {"rollout_manager": 1}, component_name=f"ROLLOUT_WORKER.{index}", **proxy_kwargs
        )
        logger = Logger(proxy.name, dump_folder=log_dir)

        """
        The event loop handles 3 types of messages from the roll-out manager:
            1)  COLLECT, upon which the agent-environment simulation will be carried out for a specified number of steps
                and the collected experiences will be sent back to the roll-out manager;
            2)  EVAL, upon which the policies contained in the message payload will be evaluated for the entire
                duration of the evaluation environment.
            3)  EXIT, upon which it will break out of the event loop and the process will terminate.

        """
        while True:
            msg = proxy.receive_once()
            if msg.tag == MsgTag.EXIT:
                logger.info("Exiting...")
                proxy.close()
                break

            if msg.tag == MsgTag.SAMPLE:
                latest = msg
                for _ in range(num_extra_recv_attempts):
                    msg = proxy.receive_once(timeout=recv_timeout)
                    if msg.body[MsgKey.EPISODE] > latest.body[MsgKey.EPISODE]:
                        logger.info(f"Skipped roll-out message for ep {latest.body[MsgKey.EPISODE]}")
                        latest = msg

                ep = latest.body[MsgKey.EPISODE]
                result = self.sample(
                    policy_state_dict=latest.body[MsgKey.POLICY_STATE], num_steps=latest.body[MsgKey.NUM_STEPS]
                )
                logger.info(
                    get_rollout_finish_msg(ep, result["step_range"], exploration_params=result["exploration_params"])
                )
                return_info = {
                    MsgKey.EPISODE: ep,
                    MsgKey.SEGMENT: latest.body[MsgKey.SEGMENT],
                    MsgKey.ROLLOUT_INFO: result["rollout_info"],
                    MsgKey.STEP_RANGE: result["step_range"],
                    MsgKey.TRACKER: result["tracker"],
                    MsgKey.END_OF_EPISODE: result["end_of_episode"]
                }
                proxy.reply(latest, tag=MsgTag.SAMPLE_DONE, body=return_info)
            elif msg.tag == MsgTag.TEST:
                tracker = self.test(msg.body[MsgKey.POLICY_STATE])
                return_info = {MsgKey.TRACKER: tracker, MsgKey.EPISODE: msg.body[MsgKey.EPISODE]}
                logger.info("Testing complete")
                proxy.reply(msg, tag=MsgTag.TEST_DONE, body=return_info)

    def actor(
        self,
        group: str,
        index: int,
        num_episodes: int,
        num_steps: int = -1,
        proxy_kwargs: dict = None,
        log_dir: str = getcwd()
    ):
        """Controller for single-threaded learning workflows.

        Args:
            group (str): Group name for the cluster that includes the server and all actors.
            index (int): Integer actor index. The actor's ID in the cluster will be "ACTOR.{actor_idx}".
            num_episodes (int): Number of training episodes. Each training episode may contain one or more
                collect-update cycles, depending on how the implementation of the roll-out manager.
            num_steps (int): Number of environment steps to roll out in each call to ``collect``. Defaults to -1, in
                which case the roll-out will be executed until the end of the environment.
            proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
                for details. Defaults to the empty dictionary.
            log_dir (str): Directory to store logs in. A ``Logger`` with tag "LOCAL_ROLLOUT_MANAGER" will be created at
                init time and this directory will be used to save the log files generated by it. Defaults to the current
                working directory.
        """
        if proxy_kwargs is None:
            proxy_kwargs = {}

        if num_steps == 0 or num_steps < -1:
            raise ValueError("num_steps must be a positive integer or -1")

        name = f"ACTOR.{index}"
        logger = Logger(name, dump_folder=log_dir)
        peers = {"policy_server": 1}
        proxy = Proxy(group, "actor", peers, component_name=name, **proxy_kwargs)
        server_address = proxy.peers["policy_server"][0]

        # get initial policy states from the policy manager
        msg = SessionMessage(MsgTag.GET_INITIAL_POLICY_STATE, proxy.name, server_address)
        reply = proxy.send(msg)[0]
        policy_state_dict, policy_version = reply.body[MsgKey.POLICY_STATE], reply.body[MsgKey.VERSION]

        # main loop
        for ep in range(1, num_episodes + 1):
            while True:
                result = self.sample(policy_state_dict=policy_state_dict, num_steps=num_steps)
                logger.info(
                    get_rollout_finish_msg(ep, result["step_range"], exploration_params=result["exploration_params"])
                )
                # Send roll-out info to policy server for learning
                reply = proxy.send(
                    SessionMessage(
                        MsgTag.SAMPLE_DONE, proxy.name, server_address,
                        body={MsgKey.ROLLOUT_INFO: result["rollout_info"], MsgKey.VERSION: policy_version}
                    )
                )[0]
                policy_state_dict, policy_version = reply.body[MsgKey.POLICY_STATE], reply.body[MsgKey.VERSION]
                if result["end_of_episode"]:
                    break

        # tell the policy server I'm all done.
        proxy.isend(SessionMessage(MsgTag.DONE, proxy.name, server_address, session_type=SessionType.NOTIFICATION))
        proxy.close()
