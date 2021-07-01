# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import defaultdict, deque

from maro.rl.experience import ExperienceSet
from maro.simulator import Env


class AbsEnvWrapper(ABC):
    """Environment wrapper that performs scenario-specific processing, transition caching and experience generation.

    Args:
        env (Env): Environment instance.
        reward_eval_delay (int): Number of ticks required after a decision event to evaluate the reward
            for the action taken for that event. Defaults to 0, which means rewards are evaluated immediately
            after executing an action.
        save_replay (bool): If True, transitions for some or all agents will in stored in internal replay buffers.
        replay_agent_ids (list): List of agent IDs whose transitions will be stored in internal replay buffers.
            If ``save_replay`` is False, this is ignored. Otherwise, if it is None, it will be set to all agents in
            Defaults to None.
    """
    def __init__(self, env: Env, reward_eval_delay: int = 0, save_replay: bool = True, replay_agent_ids: list = None):
        self.env = env
        self.state_info = None  # context for converting model output to actions that can be executed by the env
        self.reward_eval_delay = reward_eval_delay
        self.action_history = defaultdict(dict)
        self.save_replay = save_replay
        self.replay_agent_ids = self.env.agent_idx_list if not replay_agent_ids else replay_agent_ids
        self._replay_buffer = {agent_id: defaultdict(list) for agent_id in self.replay_agent_ids}
        self._pending_reward_cache = deque()  # list of (state, action, tick) whose rewards have yet to be evaluated
        self._step_index = None
        self._total_reward = defaultdict(int)
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
    def summary(self):
        return self.env.metrics, self._total_reward

    @property
    def state(self):
        """The current environmental state."""
        return self._state

    @property
    def event(self):
        return self._event

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
    def get_reward(self, tick: int = None):
        """Evaluate the reward for an action.

        Args:
            tick (int): Evaluate the reward for the action that occured at the given tick. The tick may be
                None, in which case the reward is evaluated for the latest action (i.e., immediate reward).
                Otherwise, it must be a key in the ``action_history`` attribute (i.e., there must be an action
                at that tick). Defaults to None.

        Returns:
            A dictionary with (agent ID, reward) as key-value pairs.
        """
        raise NotImplementedError

    def get_transition_info(self):
        """Get additional info for a transition.

        The returned transition info will be stored in the experience manager alongside states, actions, rewards.

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
        env_action_dict = self.to_env_action(action_by_agent)
        for agent_id, action in env_action_dict.items():
            self.action_history[self.env.tick][agent_id] = action
        transition_info = self.get_transition_info()
        self._pending_reward_cache.append((self._state, action_by_agent, transition_info, self.env.tick))

        env_action = list(env_action_dict.values())
        _, self._event, done = self.env.step(env_action)

        if not done:
            self._state = self.get_state(self.env.tick)  # current env state
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
            state, action, info, tick = self._pending_reward_cache.popleft()
            reward = self.get_reward(tick=tick)
            # assign rewards to the agents that took action at that tick
            if self.save_replay:
                for agent_id, st in state.items():
                    self._total_reward[agent_id] += reward[agent_id]
                    if agent_id in self._replay_buffer:
                        buf = self._replay_buffer[agent_id]
                        buf["states"].append(st)
                        buf["actions"].append(action[agent_id])
                        buf["rewards"].append(reward[agent_id])
                        buf["info"].append(info[agent_id] if info else None)

    def end_of_episode(self):
        """Custom processing logic at the end of an episode."""
        pass

    def get_experiences(self):
        """Get per-agent experiences from the replay buffer."""
        exp_by_agent = {}
        for agent_id in self.replay_agent_ids:
            buf = self._replay_buffer[agent_id]
            exp_set = ExperienceSet(
                buf["states"][:-1],
                buf["actions"][:-1],
                buf["rewards"][:-1],
                buf["states"][1:],
                buf["info"][:-1],
            )
            del buf["states"][:-1]
            del buf["actions"][:-1]
            del buf["rewards"][:-1]
            del buf["info"][:-1]
            exp_by_agent[agent_id] = exp_set

        return exp_by_agent

    def reset(self):
        self.env.reset()
        self.state_info = None
        self._total_reward.clear()
        self._state = None
        self._pending_reward_cache.clear()
        for replay in self._replay_buffer.values():
            replay.clear()
