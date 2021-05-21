# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict

from maro.simulator import Env
from maro.rl.experience import AbsExperienceManager, ExperienceSet


class AbsEnvWrapper(ABC):
    """Environment wrapper that performs scenario-specific processing, transition caching and experience generation.

    Args:
        env (Env): Environment instance.
        replay_buffer (dict): Replay buffers for recording transitions experienced by agents in sequential fashion.
            Transitions will only be recorded for those agents whose IDs appear in the keys of the dictionary. 
            Defaults to None, in which case no transition will be recorded.
        reward_eval_delay (int): Number of ticks required after a decision event to evaluate the reward
            for the action taken for that event. Defaults to 0, which means rewards are evaluated immediately
            after executing an action.
    """
    def __init__(
        self,
        env: Env,
        replay_buffer: Dict[str, AbsExperienceManager] = None,
        reward_eval_delay: int = 0
    ):
        self.env = env
        self.replay = replay_buffer
        self.state_info = None  # context for converting model output to actions that can be executed by the env
        self.reward_eval_delay = reward_eval_delay
        self._pending_reward_cache = deque()  # list of (state, action, tick) whose rewards have yet to be evaluated
        self._step_index = None
        self._total_reward = 0
        self._event = None  # the latest decision event. This is not used if the env wrapper is not event driven.
        self._state = None  # the latest extracted state is kept here

    @property
    def step_index(self):
        """Number of environmental steps taken so far."""
        return self._step_index
    
    @property
    def agent_idx_list(self):
        return self.env.agent_idx_list

    @property
    def metrics(self):
        return self.env.metrics

    @property
    def state(self):
        """The current environmental state."""
        return self._state

    @property
    def event(self):
        return self._event

    @property
    def total_reward(self):
        """The total reward achieved so far."""
        return self._total_reward

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
        """
        raise NotImplementedError

    @abstractmethod
    def to_env_action(self, action):
        """Convert policy outputs to an action that can be executed by ``self.env.step()``."""
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, tick: int = None):
        """Evaluate the reward for an action.

        Args:
            tick (int): If given, the reward for the action that occured at this tick will be evaluated (in the case
                of delayed reward evaluation). Otherwise, the reward is evaluated for the latest action.
                Defaults to None.
        """
        raise NotImplementedError

    def get_transition_info(self):
        """Get additional info for a transition.

        The returned transition info will be stored in the experience manager alongside states, actions, rewards.
        """
        pass

    def step(self, action_by_agent: dict):
        """Wrapper for env.step().

        The new transition is stored in the replay buffer or cached in a separate data structure if the
        reward cannot be determined yet due to a non-zero ``reward_eval_delay``.
        """
        self._step_index += 1
        env_action = self.to_env_action(action_by_agent)
        if len(env_action) == 1:
            env_action = list(env_action.values())[0]
        pre_action_tick = self.env.tick
        _, self._event, done = self.env.step(env_action)

        if not done:
            prev_state = self._state  # previous env state
            transition_info = self.get_transition_info()
            self._state = self.get_state(self.env.tick)  # current env state
            self._pending_reward_cache.append(
                (prev_state, action_by_agent, self._state, transition_info, pre_action_tick)
            )
        else:
            self._state = None
            self.end_of_episode()

        """
        If this is the final step, evaluate rewards for all remaining events except the last.
        Otherwise, evaluate rewards only for events at least self.reward_eval_delay ticks ago.
        """
        while (
            self._pending_reward_cache and
            (done or self.env.tick - self._pending_reward_cache[0][-1] >= self.reward_eval_delay)
        ):
            state, action, state_, info, tick = self._pending_reward_cache.popleft()
            reward = self.get_reward(tick=tick)
            # assign rewards to the agents that took action at that tick
            for agent_id, act in action.items():
                st, st_, rw = state[agent_id], state_[agent_id], reward.get(agent_id, .0)
                if not done and self.replay and agent_id in self.replay:
                    self.replay[agent_id].put(ExperienceSet([st], [act], [rw], [st_], [info]))
                self._total_reward += rw

    def end_of_episode(self):
        """Custom processing logic at the end of an episode."""
        pass

    def get_experiences(self):
        """Get per-agent experiences from the replay buffer."""
        return {agent_id: replay.batch() for agent_id, replay in self.replay.items()}

    def reset(self):
        self.env.reset()
        self.state_info = None
        self._total_reward = 0
        self._state = None
        self._pending_reward_cache.clear()
        for replay in self.replay.values():
            replay.clear()
