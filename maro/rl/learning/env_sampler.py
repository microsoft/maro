# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, deque
from os import getcwd
from typing import Callable, Dict

from maro.communication import Proxy, SessionMessage, SessionType
from maro.rl.policy import RLPolicy
from maro.rl.utils import MsgKey, MsgTag
from maro.simulator import Env
from maro.utils import Logger

from .common import get_rollout_finish_msg


class EnvSampler:
    """Simulation data collector and policy evaluator.

    Args:
        get_env (Callable[[], Env]): Function to create an ``Env`` instance for collecting training data. The function
            should take no parameters and return an environment wrapper instance.
        get_policy_func_dict (dict): A dictionary mapping policy names to functions that create them. The policy
            creation function should have policy name as the only parameter and return an ``AbsPolicy`` instance.
        agent2policy (Dict[str, str]): A dictionary that maps agent IDs to policy IDs, i.e., specifies the policy used
            by each agent.
        get_state (Callable): Function to compute the state. The function takes as input an ``Env``, an event and a
            dictionary of keyword parameters and returns a state vector encoded as a one-dimensional (flattened) Numpy
            arrays for each agent involved as a dictionary.
        get_env_actions (Callable): Function to convert policy outputs to action objects that can be passed directly to
            the environment's ``step`` method. The function takes as input an ``Env``, a dictionary of a set of agents'
            policy outputs, an event and a dictionary of keyword parameters and returns a list of action objects.
        get_reward (Callable): Function to compute rewards for a list of actions that occurred at a given tick. The
            function takes as input an ``Env``, a list of actions (output by ``get_env_actions``), a tick and a
            dictionary of keyword parameters and returns a scalar reward for each agent as a dictionary. 
        get_test_env (Callable): Function to create an ``Env`` instance for testing policy performance. The function
            should take no parameters and return an environment wrapper instance. If this is None, the training
            environment wrapper will be used for evaluation in the worker processes. Defaults to None.
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
        get_state: Dict[str, Callable],
        get_env_actions: Callable,
        get_reward: Callable, 
        get_test_env: Callable[[], Env] = None,
        reward_eval_delay: int = 0,
        post_step: Callable = None,
        state_shaping_kwargs: dict = {},
        action_shaping_kwargs: dict = {},
        reward_shaping_kwargs: dict = {}
    ):
        self._learn_env = get_env()
        self._test_env = get_test_env() if get_test_env else self._learn_env
        self.env = None

        self.policy_dict = {id_: func(id_) for id_, func in get_policy_func_dict.items()}
        self.policy_by_agent = {agent: self.policy_dict[policy_id] for agent, policy_id in agent2policy.items()}
        self.reward_eval_delay = reward_eval_delay
        self._post_step = post_step

        # shaping
        self._get_state = get_state
        self._state_shaping_kwargs = state_shaping_kwargs
        self._get_env_actions = get_env_actions
        self._action_shaping_kwargs = action_shaping_kwargs
        self._get_reward = get_reward
        self._reward_shaping_kwargs = reward_shaping_kwargs

        self._step_index = 0
        self._terminal = True

        self._transition_cache = defaultdict(deque)  # for caching transitions whose rewards have yet to be evaluated
        self._prev_state = defaultdict(lambda: None)

        self.tracker = {}  # User-defined tracking information is placed here.

    def sample(self, policy_state_dict: dict = None, num_steps: int = -1, exploration_step: bool = False):
        self.env = self._learn_env
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
            # reset and get initial state
            self.env.reset()
            self._step_index = 0
            self._transition_cache.clear()
            self._prev_state = defaultdict(lambda: None)
            self.tracker.clear()
            self._terminal = False
            _, event, _ = self.env.step(None)
            state = self._get_state(self.env, event, self._state_shaping_kwargs)

        starting_step_index = self._step_index + 1
        steps_to_go = float("inf") if num_steps == -1 else num_steps
        while not self._terminal and steps_to_go > 0:
            action = {agent: self.policy_by_agent[agent].choose_action(st) for agent, st in state.items()}
            env_actions = self._get_env_actions(self.env, action, event, self._action_shaping_kwargs)
            for agent in state:
                self._transition_cache[agent].append((state[agent], action[agent], env_actions, self.env.tick))
            _, event, self._terminal = self.env.step(env_actions)
            state = None if self._terminal else self._get_state(self.env, event, self._state_shaping_kwargs)
            self._step_index += 1
            steps_to_go -= 1

        """
        If this is the final step, evaluate rewards for all remaining events except the last.
        Otherwise, evaluate rewards only for events at least self.reward_eval_delay ticks ago.
        """
        for agent, cache in self._transition_cache.items():
            while cache and (self._terminal or self.env.tick - cache[0][-1] >= self.reward_eval_delay):
                state, action, env_actions, tick = cache.popleft()
                reward = self._get_reward(self.env, env_actions, tick, self._reward_shaping_kwargs)
                if self._post_step:
                    # put things you want to track in the tracker attribute
                    self._post_step(self.env, self.tracker, state, action, env_actions, reward, tick)

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
        self.env = self._test_env
        # set policy states
        if policy_state_dict:
            for id_, policy_state in policy_state_dict.items():
                self.policy_dict[id_].set_state(policy_state)

        # Set policies to exploitation mode
        for policy in self.policy_dict.values():
            if hasattr(policy, "exploit"):
                policy.exploit()

        self.env.reset()
        terminal = False
        # get initial state
        _, event, _ = self.env.step(None)
        state = self._get_state(self.env, event, self._state_shaping_kwargs)
        while not terminal:
            action = {agent: self.policy_by_agent[agent].choose_action(st) for agent, st in state.items()}
            env_actions = self._get_env_actions(self.env, action, event, self._action_shaping_kwargs)
            _, event, terminal = self.env.step(env_actions)
            if not terminal:
                state = self._get_state(self.env, event, self._state_shaping_kwargs)

        return self.tracker

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
        server_address = proxy.peers["policy_server"][0]
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
                        MsgTag.SAMPLE_DONE, proxy.name, server_address,
                        body={MsgKey.ROLLOUT_INFO: result["rollout_info"], MsgKey.VERSION: policy_version}
                    )
                )[0]
                policy_state_dict, policy_version = reply.body[MsgKey.POLICY_STATE], reply.body[MsgKey.VERSION]
                if result["end_of_episode"]:
                    break

                exploration_step = False

        # tell the policy server I'm all done.
        proxy.isend(SessionMessage(MsgTag.DONE, proxy.name, server_address, session_type=SessionType.NOTIFICATION))
        proxy.close()
