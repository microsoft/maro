# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from os import getcwd
from typing import Callable, Dict

from maro.communication import Proxy, SessionMessage, SessionType
from maro.rl.policy import AbsPolicy, RLPolicy
from maro.rl.utils import MsgKey, MsgTag
from maro.simulator import Env
from maro.utils import Logger

from .common import get_rollout_finish_msg


class AbsEnvSampler(ABC):
    """Simulation data collector and policy evaluator.

    Args:
        get_env (Callable[[], Env]): Function to create an ``Env`` instance for collecting training data. The function
            should take no parameters and return an environment wrapper instance.
        get_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have policy name as the only parameter and return an ``AbsPolicy`` instance.
        agent2policy (Dict[str, str]): A dictionary that maps agent IDs to policy IDs, i.e., specifies the policy used
            by each agent. 
        get_eval_env (Callable): Function to create an ``Env`` instance for evaluation. The function should
            take no parameters and return an environment wrapper instance. If this is None, the training environment
            wrapper will be used for evaluation in the worker processes. Defaults to None.
        reward_eval_delay (int): Number of ticks required after a decision event to evaluate the reward
            for the action taken for that event. Defaults to 0, which means rewards are evaluated immediately
            after executing an action.
        replay_agent_ids (list): List of agent IDs whose transitions will be stored in internal replay buffers.
            If it is None, it will be set to all agents in the environment (i.e., env.agent_idx_list). Defaults
            to None.
        post_step (Callable): Custom function to gather information about a transition and the evolvement of the
            environment. The function signature should be (env, tracker, transition) -> None, where env is the ``Env``
            instance in the wrapper, tracker is a dictionary where the gathered information is stored and transition
            is a ``Transition`` object. For example, this callback can be used to collect various statistics on the
            simulation. Defaults to None.
    """
    def __init__(
        self,
        get_env: Callable[[], Env],
        get_policy_func_dict: Dict[str, Callable],
        agent2policy: Dict[str, str],
        get_eval_env: Callable[[], Env] = None,
        reward_eval_delay: int = 0,
        post_step: Callable = None
    ):
        self.env = get_env()
        self.policy_dict = {id_: func(id_) for id_, func in get_policy_func_dict.items()}
        self.policy_by_agent = {agent: self.policy_dict[policy_id] for agent, policy_id in agent2policy.items()}
        self.eval_env = get_eval_env() if get_eval_env else self.env
        self.reward_eval_delay = reward_eval_delay
        self._post_step = post_step

        self._step_index = 0
        self._terminal = True

        self._transition_cache = defaultdict(deque)  # for caching transitions whose rewards have yet to be evaluated
        self._prev_state = defaultdict(lambda: None)        

        self.tracker = {}  # User-defined tracking information is placed here.

    @abstractmethod
    def get_state(self, event, eval: bool = False, tick: int = None) -> dict:
        """Compute the state for a given tick.

        Args:
            tick (int): The tick for which to compute the environmental state. If computing the current state,
                use tick=self.env.tick.

        Returns:
            A dictionary with (agent ID, state) as key-value pairs.
        """
        raise NotImplementedError

    @abstractmethod
    def to_env_action(self, action, eval: bool = False) -> dict:
        """Convert policy outputs to an action that can be executed by ``self.env.step()``."""
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, actions: list, tick: int = None, eval: bool = False):
        """Evaluate the reward for an action.

        Args:
            tick (int): Evaluate the reward for the actions that occured at the given tick. Each action in
                ``actions`` must be an Action object defined for the environment in question. The tick may
                be None, in which case the reward is evaluated for the latest action (i.e., immediate reward).
                Defaults to None.

        Returns:
            A dictionary with (agent ID, reward) as key-value pairs.
        """
        raise NotImplementedError

    def get_agents_from_event(self, event):
        return self.env.agent_idx_list

    def sample(self, policy_state_dict: dict = None, num_steps: int = -1, exploration_step: bool = False):
        # set policy states
        if policy_state_dict:
            for policy_id, policy_state in policy_state_dict.items():
                self.policy_dict[policy_id].set_state(policy_state)

        # update exploration states if necessary
        for policy in self.policy_dict.values():
            if hasattr(policy, "explore"):
                policy.explore()

        if exploration_step:
            for policy in self.policy_dict.values():
                if hasattr(policy, "exploration_step"):
                    policy.exploration_step()

        if self._terminal:
            # get initial state
            self.reset()
            _, event, _ = self.env.step(None)
            self._state = self.get_state(event)

        starting_step_index = self._step_index + 1
        steps_to_go = float("inf") if num_steps == -1 else num_steps
        while not self._terminal and steps_to_go > 0:
            action = {agent: self.policy_by_agent[agent].choose_action(st) for agent, st in self._state.items()}
            env_action = self.to_env_action(action)
            for agent in self._state:
                self._transition_cache[agent].append((self._state[agent], action[agent], env_action, self.env.tick))
            _, event, self._terminal = self.env.step(env_action)
            self._state = None if self._terminal else self.get_state(event)
            self._step_index += 1
            steps_to_go -= 1

        """
        If this is the final step, evaluate rewards for all remaining events except the last.
        Otherwise, evaluate rewards only for events at least self.reward_eval_delay ticks ago.
        """
        for agent, cache in self._transition_cache.items():
            while cache and (self._terminal or self.env.tick - cache[0][-1] >= self.reward_eval_delay):
                state, action, env_action, tick = cache.popleft()
                reward = self.get_reward(env_action, tick=tick)
                if self._post_step:
                    # put things you want to track in the tracker attribute
                    self._post_step(self.env, self.tracker, state, action, env_action, reward, tick)

                if isinstance(self.policy_by_agent[agent], RLPolicy) and self._prev_state[agent] is not None:
                    self.policy_by_agent[agent].record(
                        agent, self._prev_state[agent], action, reward[agent], state, not cache and self._terminal
                    )
                self._prev_state[agent] = state

        return {
            "rollout_info": {
                id_: policy.get_rollout_info() for id_, policy in self.policy_dict.items()
                if isinstance(policy, RLPolicy)
            },
            "step_range": (starting_step_index, self._step_index),
            "tracker": self.tracker,
            "end_of_episode": self._terminal,
            "exploration_params": {
                name: policy.exploration_params for name, policy in self.policy_dict.items()
                if isinstance(policy, RLPolicy)
            }
        }

    def test(self, policy_state_dict: dict = None):
        # set policy states
        if policy_state_dict:
            for id_, policy_state in policy_state_dict.items():
                self.policy_dict[id_].set_state(policy_state)

        # Set policies to exploitation mode
        for policy in self.policy_dict.values():
            if hasattr(policy, "exploit"):
                policy.exploit()

        self.eval_env.reset()
        terminal = False
        # get initial state
        _, event, _ = self.eval_env.step(None)
        state = self.get_state(event, eval=True)
        while not terminal:
            action = {agent: self.policy_by_agent[agent].choose_action(st) for agent, st in state.items()} 
            env_action = self.to_env_action(action, eval=True)
            _, _, terminal = self.eval_env.step(env_action)
            if not terminal:
                state = self.get_state(event, eval=True)

        return self.tracker

    def reset(self):
        self.env.reset()
        self._step_index = 0
        self._state = None
        self._transition_cache.clear()
        self.tracker.clear()
        self._terminal = False

    def worker(self, group: str, index: int, proxy_kwargs: dict = {}, log_dir: str = getcwd()):
        """Roll-out worker process that can be launched on separate computation nodes.

        Args:
            group (str): Group name for the roll-out cluster, which includes all roll-out workers and a roll-out manager
                that manages them.
            worker_idx (int): Worker index. The worker's ID in the cluster will be "ROLLOUT_WORKER.{worker_idx}".
                This is used for bookkeeping by the parent manager.
            proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
                for details. Defaults to the empty dictionary.
            log_dir (str): Directory to store logs in. Defaults to the current working directory.
        """
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
        for msg in proxy.receive():
            if msg.tag == MsgTag.EXIT:
                logger.info("Exiting...")
                proxy.close()
                break

            if msg.tag == MsgTag.SAMPLE:
                ep = msg.body[MsgKey.EPISODE]
                result = self.sample(
                    policy_state_dict=msg.body[MsgKey.POLICY_STATE],
                    num_steps=msg.body[MsgKey.NUM_STEPS],
                    exploration_step=msg.body[MsgKey.EXPLORATION_STEP]
                )
                logger.info(
                    get_rollout_finish_msg(ep, result["step_range"], exploration_params=result["exploration_params"])
                )
                return_info = {
                    MsgKey.EPISODE: ep,
                    MsgKey.SEGMENT: msg.body[MsgKey.SEGMENT],
                    MsgKey.VERSION: msg.body[MsgKey.VERSION],
                    MsgKey.ROLLOUT_INFO: result["rollout_info"],
                    MsgKey.STEP_RANGE: result["step_range"],
                    MsgKey.TRACKER: result["tracker"],
                    MsgKey.END_OF_EPISODE: result["end_of_episode"]
                }
                proxy.reply(msg, tag=MsgTag.SAMPLE_DONE, body=return_info)
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
        if num_steps == 0 or num_steps < -1:
            raise ValueError("num_steps must be a positive integer or -1")

        peers = {"policy_server": 1}
        proxy = Proxy(group, "actor", peers, component_name=f"ACTOR.{index}", **proxy_kwargs)
        policy_server_address = proxy.peers["policy_server"][0]
        logger = Logger(proxy.name, dump_folder=log_dir)

        # get initial policy states from the policy manager
        msg = SessionMessage(MsgTag.GET_INITIAL_POLICY_STATE, proxy.name, policy_server_address)
        reply = proxy.send(msg)[0]
        policy_state_dict, policy_version = reply.body[MsgKey.POLICY_STATE], reply.body[MsgKey.VERSION]

        # main loop
        for ep in range(1, num_episodes + 1):
            exploration_step = True
            while True:
                result = self.sample(
                    policy_state_dict=policy_state_dict, num_steps=num_steps, exploration_step=exploration_step
                )
                logger.info(
                    get_rollout_finish_msg(ep, result["step_range"], exploration_params=result["exploration_params"])
                )
                # Send roll-out info to policy server for learning
                reply = proxy.send(
                    SessionMessage(
                        MsgTag.SAMPLE_DONE, proxy.name, policy_server_address,
                        body={MsgKey.ROLLOUT_INFO: result["rollout_info"], MsgKey.VERSION: policy_version}
                    )
                )[0]
                policy_state_dict, policy_version = reply.body[MsgKey.POLICY_STATE], reply.body[MsgKey.VERSION]
                if result["end_of_episode"]:
                    break

                exploration_step = False

        # tell the policy server I'm all done.
        proxy.isend(
            SessionMessage(MsgTag.DONE, proxy.name, policy_server_address, session_type=SessionType.NOTIFICATION)
        )
        proxy.close()
