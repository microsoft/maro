# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from collections import defaultdict, deque

from maro.rl import AbsEnvWrapper
from maro.simulator import Env

from .components import VMAction, VMState


class VMEnvWrapper(AbsEnvWrapper):
    def __init__(
        self,
        env: Env,
        training: bool,
        alpha: float,
        beta: float,
        pm_num: int,
        durations: int,
        vm_window_size: int = 1,
        pm_window_size: int = 1,
        gamma: float = 0.0,
        reward_eval_delay: int = 0,
        save_replay: bool = True
    ):
        self.env = env
        self.state_info = None  # context for converting model output to actions that can be executed by the env
        self.reward_eval_delay = reward_eval_delay
        self.save_replay = save_replay

        self._replay_buffer = defaultdict(list)
        self.action_history = dict()

        self._pending_reward_cache = deque()  # list of (state, action, tick) whose rewards have yet to be evaluated
        self._step_index = None
        self._total_reward = 0.0
        self._event = None  # the latest decision event. This is not used if the env wrapper is not event driven
        self._state = None  # the latest extracted state is kept here

        self._legal_pm = None
        self.action_class = VMAction(pm_num) # convert the action info to the environment action
        self.state_class = VMState(pm_num, durations, training, vm_window_size, pm_window_size) # get the state information

        self._training = training
        self._alpha, self._beta = alpha, beta # adjust the ratio of the success allocation and the total income when computing the reward
        self._gamma = gamma # reward discount
        self._pm_num = pm_num # the number of pms
        self._durations = durations # the duration of the whole environment

    def start(self):
        """Generate the initial environmental state at the beginning of a simulation episode."""
        self._step_index = 0
        _, self._event, _ = self.env.step(None)
        self._state, self._legal_pm = self.get_state()

    def get_state(self):
        return self.state_class(self.env, self._event)

    @property
    def event(self):
        return self._event

    @property
    def legal_pm(self):
        return self._legal_pm

    def to_env_action(self, action_info):
        model_action = action_info[0] if isinstance(action_info, tuple) else action_info
        return self.action_class(model_action, self._event)

    def get_reward(self, action_info):
        model_action = action_info[0] if isinstance(action_info, tuple) else action_info
        if model_action == self._pm_num:
            if np.sum(self._legal_pm) != 1:
                reward = -0.1 * self._alpha + 0.0 * self._beta
            else:
                reward = 0.0 * self._alpha + 0.0 * self._beta
        else:
            reward = (
                1.0 * self._alpha
                + (
                    self._event.vm_unit_price
                    * min(self._durations - self._event.frame_index, self._event.vm_lifetime)
                ) * self._beta
            )
        return reward

    def step(self, action_info):
        """Wrapper for env.step().
        The new transition is stored in the replay buffer or cached in a separate data structure if the
        reward cannot be determined yet due to a non-zero ``reward_eval_delay``.
        """
        self._step_index += 1
        env_action = self.to_env_action(action_info)
        self.action_history[self.env.tick] = env_action

        reward = self.get_reward(action_info)
        self._total_reward += reward

        if self.save_replay:
            buf = self._replay_buffer
            buf["states"].append(self._state)
            buf["actions"].append(action_info)
            buf["rewards"].append(reward)
            buf["info"].append(self._legal_pm)

        _, self._event, done = self.env.step(env_action)

        if not done:
            self._state, self._legal_pm = self.get_state()  # current env state
        else:
            self._state, self._legal_pm = None, None
            self.end_of_episode()

    def reset(self):
        self.env.reset()
        self.state_info = None
        self._total_reward = 0.0
        self._state = None
        self._legal_pm = None
        self._pending_reward_cache.clear()
        self._replay_buffer.clear()
