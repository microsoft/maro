# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Callable

from maro.simulator import Env
from maro.rl.types import Trajectory, Transition


class AbsEnvWrapper(ABC):
    """Environment wrapper that performs scenario-specific processing, transition caching and experience generation.

    Args:
        env (Env): Environment instance.
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
        env: Env,
        reward_eval_delay: int = 0,
        replay_agent_ids: list = None,
        post_step: Callable = None
    ):
        self.env = env
        self.reward_eval_delay = reward_eval_delay

        self._post_step = post_step

        replay_agent_ids = self.env.agent_idx_list if not replay_agent_ids else replay_agent_ids
        self._replay_buffer = {agent_id: defaultdict(list) for agent_id in replay_agent_ids}
        self._transition_cache = deque()  # list of (state, action, tick) whose rewards have yet to be evaluated
        self._step_index = None
        self._event = None  # the latest decision event. This is not used if the env wrapper is not event driven.
        self._state = None  # the latest extracted state is kept here

        self.tracker = {}  # User-defined tracking information is placed here.
        self._replay = True

    @property
    def step_index(self):
        """Number of environmental steps taken so far."""
        return self._step_index

    @property
    def agent_idx_list(self):
        return self.env.agent_idx_list

    @property
    def summary(self):
        return self.env.metrics

    @property
    def state(self):
        """The current environmental state."""
        return self._state

    @property
    def event(self):
        return self._event

    def collect(self):
        self._replay = True

    def evaluate(self):
        self._replay = False

    def start(self):
        """Generate the initial environmental state at the beginning of a simulation episode."""
        self._step_index = 0
        _, self._event, _ = self.env.step(None)
        self._state = self.get_state(self.env.tick)

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
    def to_env_action(self, action) -> dict:
        """Convert policy outputs to an action that can be executed by ``self.env.step()``."""
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, actions: list, tick: int = None):
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

    def get_transition_info(self, tick: int = None):
        """Get additional info for a transition.

        The returned transition info will be stored in the experience manager alongside states, actions, rewards.

        Args:
            tick (int): The tick for which to compute the environmental state. If computing the current state,
                use tick=self.env.tick.

        Returns:
            A dictionary with (agent ID, transition_info) as key-value pairs.

        """
        pass

    def step(self, action_by_agent: dict):
        """Wrapper for env.step().

        The new transition is stored in the replay buffer or cached in a separate data structure if the
        reward cannot be determined yet due to a non-zero ``reward_eval_delay``.
        """
        self._step_index += 1
        env_action = self.to_env_action(action_by_agent)

        self._transition_cache.append((
            self._state,
            action_by_agent,
            env_action,
            self.get_transition_info(),
            self.env.tick
        ))

        _, self._event, done = self.env.step(env_action)

        if not done:
            self._state = self.get_state(self.env.tick)  # current env state
        else:
            self._state = None

        """
        If this is the final step, evaluate rewards for all remaining events except the last.
        Otherwise, evaluate rewards only for events at least self.reward_eval_delay ticks ago.
        """
        while (
            self._transition_cache and
            (done or self.env.tick - self._transition_cache[0][-1] >= self.reward_eval_delay)
        ):
            state, action, env_action, info, tick = self._transition_cache.popleft()
            reward = self.get_reward(env_action, tick=tick)
            if self._post_step:
                next_state = self._transition_cache[0][0] if self._transition_cache else None
                transition = Transition(state, action, env_action, reward, next_state, info)
                # put things you want to track in the tracker attribute
                self._post_step(self.env, self.tracker, transition)

            if self._replay:
                for agent_id, agent_state in state.items():
                    if agent_id in self._replay_buffer:
                        buf = self._replay_buffer[agent_id]
                        buf["states"].append(agent_state)
                        buf["actions"].append(action[agent_id])
                        buf["rewards"].append(reward[agent_id])
                        buf["info"].append(info[agent_id] if info else None)

    def get_trajectory(self, clear_buffer: bool = True):
        """Get agent trajectories from the replay buffer and clear it in the process."""
        trajectory = {}
        for agent_id, buf in self._replay_buffer.items():
            # end of an episode
            if not self._state:
                states = buf["states"] + [None]
                actions = buf["actions"] + [None]
                rewards = buf["rewards"][:]
                info = buf["info"][:]
                if clear_buffer:
                    del buf["states"]
                    del buf["actions"]
                    del buf["rewards"]
                    del buf["info"]
            else:
                states = buf["states"][:]
                actions = buf["actions"][:] 
                rewards = buf["rewards"][:-1]
                info = buf["info"][:-1]
                if clear_buffer:
                    del buf["states"][:-1]
                    del buf["actions"][:-1]
                    del buf["rewards"][:-1]
                    del buf["info"][:-1]

            trajectory[agent_id] = Trajectory(states, actions, rewards, info)            

        return trajectory

    def reset(self):
        self.env.reset()
        self._state = None
        self._transition_cache.clear()
        self.tracker.clear()
        for replay in self._replay_buffer.values():
            replay.clear()
