# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from multiprocessing import Pipe, Process
from typing import Callable, Dict

import numpy as np

from maro.communication import Proxy, SessionMessage, SessionType
from maro.rl.policy import RLPolicy
from maro.rl.policy_v2 import AbsRLPolicy
from maro.rl.utils import MsgKey, MsgTag
from maro.simulator import Env
from maro.utils import DummyLogger, Logger

from .helpers import get_rollout_finish_msg


class AbsAgentWrapper:
    def __init__(self, agent2policy: Dict[str, str]) -> None:
        self._agent2policy = agent2policy

    @abstractmethod
    def load(self, dir: str) -> None:
        pass

    @abstractmethod
    def choose_action(self, states_by_agent: Dict[str, np.ndarray]) -> dict:
        pass

    @abstractmethod
    def set_policy_states(self, policy_state_dict: dict) -> None:
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
    def improve(self) -> None:
        pass

    @abstractmethod
    def save(self, checkpoint_dir: str) -> None:
        pass


class SimpleAgentWrapper(AbsAgentWrapper):
    """Wrapper for multiple agents using multiple policies to expose simple single-agent interfaces."""
    def __init__(self, get_policy_func_dict: Dict[str, Callable], agent2policy: Dict[str, str]):
        super().__init__(agent2policy)
        self.policy_dict = {policy_id: func(policy_id) for policy_id, func in get_policy_func_dict.items()}
        self.policy_by_agent = {agent: self.policy_dict[policy_id] for agent, policy_id in agent2policy.items()}
        self._rl_policy_dict = {
            id_: policy for id_, policy in self.policy_dict.items() if isinstance(policy, (RLPolicy, AbsRLPolicy))
        }

    def load(self, dir: str):
        for id_, policy in self.policy_dict.items():
            pth = os.path.join(dir, id_)
            if os.path.exists(pth):
                policy.load(pth)

    def choose_action(self, state_by_agent: Dict[str, np.ndarray]):
        states_by_policy, agents_by_policy = defaultdict(list), defaultdict(list)
        for agent, state in state_by_agent.items():
            states_by_policy[self._agent2policy[agent]].append(state)
            agents_by_policy[self._agent2policy[agent]].append(agent)

        action_by_agent = {}
        # compute the actions for local policies first while the inferences processes do their work.
        for policy_id, policy in self.policy_dict.items():
            if states_by_policy[policy_id]:
                action_by_agent.update(
                    zip(agents_by_policy[policy_id], policy(np.vstack(states_by_policy[policy_id])))
                )

        return action_by_agent

    def set_policy_states(self, policy_state_dict: dict):
        for policy_id, policy_state in policy_state_dict.items():
            self.policy_dict[policy_id].set_state(policy_state)

    def explore(self):
        for policy in self._rl_policy_dict.values():
            policy.explore()

    def exploit(self):
        for policy in self._rl_policy_dict.values():
            policy.exploit()

    def exploration_step(self):
        for policy in self._rl_policy_dict.values():
            policy.exploration_step()

    def get_rollout_info(self):
        return {id_: policy.get_rollout_info() for id_, policy in self._rl_policy_dict.items()}

    def get_exploration_params(self):
        return {id_: policy.get_exploration_params() for id_, policy in self._rl_policy_dict.items()}

    def record_transition(
        self, agent: str, state: np.ndarray, action: dict, reward: float, next_state: np.ndarray, terminal: bool
    ):
        if isinstance(self.policy_by_agent[agent], (RLPolicy, AbsRLPolicy)):
            self.policy_by_agent[agent].record(agent, state, action, reward, next_state, terminal)

    def improve(self):
        for policy in self._rl_policy_dict.values():
            policy.improve()

    def save(self, checkpoint_dir: str):
        for id_, policy in self._rl_policy_dict.items():
            policy.save(os.path.join(checkpoint_dir, id_))


class ParallelAgentWrapper(AbsAgentWrapper):
    """Wrapper for multiple agents using multiple policies to expose simple single-agent interfaces.

    The policy instances are distributed across multiple processes to achieve parallel inference.
    """
    def __init__(self, get_policy_func_dict: Dict[str, Callable], agent2policy: Dict[str, str]):
        super().__init__(agent2policy)
        self._inference_services = []
        self._conn = {}
        self._rl_conn = {}

        def _inference_service(id_, get_policy, conn):
            policy = get_policy(id_)
            conn.send(isinstance(policy, (RLPolicy, AbsRLPolicy)))
            while True:
                msg = conn.recv()
                conn.send(getattr(policy, msg["type"])(*msg.get("args", ())))

        for policy_id in get_policy_func_dict:
            conn1, conn2 = Pipe()
            self._conn[policy_id] = conn1
            host = Process(
                target=_inference_service,
                args=(policy_id, get_policy_func_dict[policy_id], conn2),
                daemon=True
            )
            self._inference_services.append(host)
            host.start()
            if conn1.recv():
                self._rl_conn[policy_id] = conn1

    def load(self, dir: str):
        for id_, conn in self._rl_conn.items():
            conn.send({"type": "load", "args": (os.path.join(dir, id_),)})
        for conn in self._rl_conn.values():
            conn.recv()

    def choose_action(self, state_by_agent: Dict[str, np.ndarray]):
        states_by_policy, agents_by_policy = defaultdict(list), defaultdict(list)
        for agent, state in state_by_agent.items():
            states_by_policy[self._agent2policy[agent]].append(state)
            agents_by_policy[self._agent2policy[agent]].append(agent)

        # send state batch to inference processes for parallelized inference.
        for policy_id, conn in self._conn.items():
            if states_by_policy[policy_id]:
                conn.send({"type": "__call__", "args": (np.vstack(states_by_policy[policy_id]),)})

        action_by_agent = {}
        for policy_id, conn in self._conn.items():
            if states_by_policy[policy_id]:
                action_by_agent.update(zip(agents_by_policy[policy_id], conn.recv()))

        return action_by_agent

    def set_policy_states(self, policy_state_dict: dict):
        for policy_id, conn in self._rl_conn.items():
            conn.send({"type": "set_state", "args": (policy_state_dict[policy_id],)})
        for conn in self._rl_conn.values():
            conn.recv()

    def explore(self):
        for conn in self._rl_conn.values():
            conn.send({"type": "explore"})
        for conn in self._rl_conn.values():
            conn.recv()

    def exploit(self):
        for conn in self._rl_conn.values():
            conn.send({"type": "exploit"})
        for conn in self._rl_conn.values():
            conn.recv()

    def exploration_step(self):
        for conn in self._rl_conn.values():
            conn.send({"type": "exploration_step"})
        for conn in self._rl_conn.values():
            conn.recv()

    def get_rollout_info(self):
        rollout_info = {}
        for conn in self._rl_conn.values():
            conn.send({"type": "rollout_info"})

        for policy_id, conn in self._rl_conn.items():
            rollout_info[policy_id] = conn.recv()

        return rollout_info

    def get_exploration_params(self):
        exploration_params = {}
        for conn in self._rl_conn.values():
            conn.send({"type": "get_exploration_params"})

        for policy_id, conn in self._rl_conn.items():
            params = conn.recv()
            if params:
                exploration_params[policy_id] = params

        return exploration_params

    def record_transition(self, agent: str, state, action, reward, next_state, terminal: bool):
        self._conn[self._agent2policy[agent]].send(
            {"type": "record", "args": (agent, state, action, reward, next_state, terminal)}
        )
        self._conn[self._agent2policy[agent]].recv()

    def improve(self):
        for conn in self._rl_conn.values():
            conn.send({"type": "improve"})
        for conn in self._rl_conn.values():
            conn.recv()

    def save(self, checkpoint_dir: str):
        for id_, conn in self._rl_conn.items():
            conn.send({"type": "save", "args": (os.path.join(checkpoint_dir, id_),)})
        for conn in self._rl_conn.values():
            conn.recv()


class SimpleMultiAgentWrapper(SimpleAgentWrapper):
    def __init__(self, get_policy_func_dict: Dict[str, Callable], agent2policy: Dict[str, str]):
        super(SimpleMultiAgentWrapper, self).__init__(get_policy_func_dict, agent2policy)

    def choose_action(self, state_by_agent: Dict[str, np.ndarray]):
        """Choose actions for all agents based on the observed states.
        Specially, the multi-agent policy takes list of all agents' states as input.

        Args:
            state_by_agent (dict): Observed state of all agents.

        Return:
            action_by_agent (dict): Action taken by each agent.
        """
        states_by_policy, agents_by_policy = defaultdict(list), defaultdict(list)
        for agent, state in state_by_agent.items():
            states_by_policy[self._agent2policy[agent]].append(state)
            agents_by_policy[self._agent2policy[agent]].append(agent)

        action_by_agent = {}
        # compute the actions for all local agents in multi-agent policy at the same time
        for policy_id, policy in self.policy_dict.items():
            if states_by_policy[policy_id]:
                # assume the MA policy take (state, agent_id) tuple as input, and return list of actions
                actions = policy(states_by_policy[policy_id], agents_by_policy[policy_id])
                action_by_agent.update(
                    zip(agents_by_policy[policy_id], actions)
                )

        return action_by_agent


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
        parallel_inference: bool = False,
        multi_agent_policy: bool = False
    ):
        self._learn_env = get_env()
        self._test_env = get_test_env() if get_test_env else self._learn_env
        self.env = None

        if multi_agent_policy:
            agent_wrapper_cls = SimpleMultiAgentWrapper
        else:
            agent_wrapper_cls = ParallelAgentWrapper if parallel_inference else SimpleAgentWrapper
        self.agent_wrapper: AbsAgentWrapper = agent_wrapper_cls(get_policy_func_dict, agent2policy)

        self.reward_eval_delay = reward_eval_delay
        self._state = None
        self._step_index = 0

        self._transition_cache = defaultdict(deque)  # for caching transitions whose rewards have yet to be evaluated
        self.tracker = {}  # User-defined tracking information is placed here.

    @abstractmethod
    def get_state(self, event, tick: int = None) -> dict:
        """Extract state for an event.

        It may be necessary to obtain extra information from the ``env`` attribute, i.e., ``self.env`` to construct
        the state required for your algorithm in addition to the information contained in ``event``.

        Args:
            event: The decision event that prompts an action.
            tick (int): The tick at which the event occurred. If computing the current state, this may be ignored as
                you can always get the current tick from ``self.env.tick()``.

        Returns:
            A dictionary of (agent ID, state) that contains the constructed states for the agents in question.
        """
        raise NotImplementedError

    @abstractmethod
    def get_env_action(self, action: dict, event) -> dict:
        """Convert policy outputs to an action that can be executed by ``self.env.step()``.

        Args:
            action (dict): A dictionary of (agent ID, policy output) where "policy output" is the output of the policy
                to which the agent is mapped.
            event: The decision event to which the action corresponds

        Returns:
            A dictionary of (agent ID, action obj) where "action obj" is an action object that can be executed by
                the environment, i.e., ``env.step()``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, env_action: dict, tick: int) -> Dict[str, float]:
        """Evaluate the reward for an action.

        Args:
            env_action: The action to assign rewards to. This should be the result of a previous call to
                ``get_env_action`` and contain actual action objects.
            tick (int): The tick at which ``env_action`` was taken. This is needed for delayed reward evaluation.
                If computing the current state, this may be ignored as you can always get the current tick from
                ``self.env.tick()``.

        Returns:
            A dictionary of (agent ID, reward) that contains the scalar rewards for the agents involved in
                ``env_action``.
        """
        raise NotImplementedError

    def sample(self, policy_state_dict: dict = None, num_steps: int = -1, return_rollout_info: bool = True):
        self.env = self._learn_env
        if not self._state:
            # reset and get initial state
            self.env.reset()
            self._step_index = 0
            self._transition_cache.clear()
            self.tracker.clear()
            _, event, _ = self.env.step(None)
            self._state = self.get_state(event)

        # set policy states
        if policy_state_dict:
            self.agent_wrapper.set_policy_states(policy_state_dict)
        self.agent_wrapper.explore()

        starting_step_index = self._step_index + 1
        steps_to_go = float("inf") if num_steps == -1 else num_steps
        while self._state and steps_to_go > 0:
            action = self.agent_wrapper.choose_action(self._state)
            env_action = self.get_env_action(action, event)
            for agent, state in self._state.items():
                self._transition_cache[agent].append((state, action[agent], env_action, self.env.tick))
            _, event, done = self.env.step(list(env_action.values()))
            self._state = None if done else self.get_state(event)
            self._step_index += 1
            steps_to_go -= 1

        """
        If this is the final step, evaluate rewards for all remaining events except the last.
        Otherwise, evaluate rewards only for events at least self.reward_eval_delay ticks ago.
        """
        for agent, cache in self._transition_cache.items():
            while cache and (not self._state or self.env.tick - cache[0][-1] >= self.reward_eval_delay):
                state, action, env_action, tick = cache.popleft()
                reward = self.get_reward(env_action, tick)
                self.post_step(state, action, env_action, reward, tick)
                self.agent_wrapper.record_transition(
                    agent, state, action, reward[agent], cache[0][0] if cache else self._state,
                    not cache and not self._state
                )

        result = {
            "step_range": (starting_step_index, self._step_index),
            "tracker": self.tracker,
            "end_of_episode": not self._state,
            "exploration_params": self.agent_wrapper.get_exploration_params()
        }
        if return_rollout_info:
            result["rollout_info"] = self.agent_wrapper.get_rollout_info()

        if not self._state:
            self.agent_wrapper.exploration_step()
        return result

    def test(self, policy_state_dict: dict = None):
        self.env = self._test_env
        # set policy states
        if policy_state_dict:
            self.agent_wrapper.set_policy_states(policy_state_dict)

        # Set policies to exploitation mode
        self.agent_wrapper.exploit()

        self.env.reset()
        terminal = False
        # get initial state
        _, event, _ = self.env.step(None)
        state = self.get_state(event)
        while not terminal:
            action = self.agent_wrapper.choose_action(state)
            env_action = self.get_env_action(action, event)
            _, event, terminal = self.env.step(list(env_action.values()))
            if not terminal:
                state = self.get_state(event)

        return self.tracker

    def post_step(self, state, action, env_actions, reward, tick):
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
        proxy_kwargs: dict = {},
        logger: Logger = DummyLogger()
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
            logger (Logger): A ``Logger`` instance for logging important events. Defaults to a ``DummyLogger``
                which logs nothing.
        """
        proxy = Proxy(
            group, "rollout_worker", {"rollout_manager": 1}, component_name=f"ROLLOUT_WORKER.{index}", logger=logger,
            **proxy_kwargs
        )
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
        proxy_kwargs: dict = {},
        logger: Logger = DummyLogger()
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
            logger (Logger): A ``Logger`` instance for logging important events. Defaults to a ``DummyLogger``
                which logs nothing.
        """
        if num_steps == 0 or num_steps < -1:
            raise ValueError("num_steps must be a positive integer or -1")

        name = f"ACTOR.{index}"
        peers = {"policy_server": 1}
        proxy = Proxy(group, "actor", peers, component_name=name, logger=logger, **proxy_kwargs)
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
