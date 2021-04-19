# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import List, Union

from maro.simulator import Env


class AbsEnvWrapper(ABC):
    """Environment wrapper that performs various shaping and other roll-out related logic.

    Args:
        env (Env): Environment instance.
        save_replay (bool): If True, the steps during roll-out will be recorded sequentially. This
            includes states, actions and rewards. The decision events themselves will also be recorded
            for delayed reward evaluation purposes. Defaults to True.
        reward_eval_delay (int): Number of ticks required after a decision event to evaluate the reward
            for the action taken for that event. Defaults to 0, which rewards are evaluated immediately
            after executing an action.
    """
    def __init__(self, env: Env, save_replay: bool = True, reward_eval_delay: int = 0):
        self.replay = defaultdict(lambda: defaultdict(list))
        self.state_info = None  # context for converting model output to actions that can be executed by the env
        self.save_replay = save_replay
        self.reward_eval_delay = reward_eval_delay
        self.pending_reward_ticks = deque()  # list of ticks whose actions have not been given a reward
        self.action_history = {}   # store the tick-to-action mapping
        self._step_index = None
        self._total_reward = 0
        self._event = None  # the latest decision event. This is not used if the env wrapper is not event driven.
        self._state = None  # the latest extracted state is kept here

    @property
    def step_index(self):
        return self._step_index
    
    @property
    def agent_idx_list(self):
        return self.env.agent_idx_list

    def start(self, rollout_index: int = None):
        self._step_index = 0
        _, self._event, _ = self.env.step(None)
        self._state = self.get_state(self.env.tick)
        if self.save_replay:
            for agent_id, state in self._state.items():
                self.replay[agent_id]["S"].append(state)

    @property
    def metrics(self):
        return self.env.metrics

    @property
    def state(self):
        return self._state

    @property
    def event(self):
        return self._event

    @property
    def total_reward(self):
        return self._total_reward

    @abstractmethod
    def get_state(self, tick: int = None) -> dict:
        """Compute the state for a given tick.

        Args:
            tick (int): The tick for which to compute the environmental state. If computing the current state,
                use tick=self.env.tick.
        """
        pass

    @abstractmethod
    def get_action(self, action) -> dict:
        pass

    @abstractmethod
    def get_reward(self, tick: int = None) -> dict:
        """User-defined reward evaluation.

        Args:
            tick (int): If given, the action that occured at this tick will be evaluated (useful for delayed
                reward evaluation). Otherwise, the reward is evaluated for the latest action. Defaults to None.
        """
        pass

    def step(self, action_by_agent: dict):
        # t0 = time.time()
        self._step_index += 1
        env_action = self.get_action(action_by_agent)
        self.pending_reward_ticks.append(self.env.tick)
        self.action_history[self.env.tick] = env_action
        if len(env_action) == 1:
            env_action = list(env_action.values())[0]
        # t1 = time.time()
        _, self._event, done = self.env.step(env_action)
        # t2 = time.time()
        # self._tot_raw_step_time += t2 - t1

        if self.save_replay:
            for agent_id, action in action_by_agent.items():
                self.replay[agent_id]["A"].append(action)
            """
            If roll-out is complete, evaluate rewards for all remaining events except the last.
            Otherwise, evaluate rewards only for events at least self.reward_eval_delay ticks ago.
            """
            while (
                self.pending_reward_ticks and
                (done or self.env.tick - self.pending_reward_ticks[0] >= self.reward_eval_delay)
            ):
                tick = self.pending_reward_ticks.popleft()
                reward = self.get_reward(tick=tick)
                # assign rewards to the agents that took action at that tick
                for agent_id in self.action_history[tick]:
                    rw = reward.get(agent_id, 0)
                    self.replay[agent_id]["R"].append(rw)
                    self._total_reward += rw

        if not done:
            self._state = self.get_state(self.env.tick)
            if self.save_replay:
                for agent_id, state in self._state.items():
                    replay = self.replay[agent_id]
                    if replay["S"]:
                        replay["S_"].append(state)
                    replay["S"].append(state)
                    assert len(replay["S_"]) == len(replay["A"]) == len(replay["S"]) - 1

            # t3 = time.time()
            # self._tot_step_time += t3 - t0
        else:
            self._state = None
            self.post_process()
            self.end_ep_callback()

        # print(f"total raw step time: {self._tot_raw_step_time}")
        # print(f"total step time: {self._tot_step_time}")
        # self._tot_raw_step_time = 0
        # self._tot_step_time = 0

    def post_process(self):
        num_experiences = 0
        for agent_id, replay in self.replay.items():
            num_complete = min(len(replay["R"]), len(replay["S_"]))
            num_experiences += num_complete
            for k, vals in replay.items():
                del vals[num_complete:]

        return experience, num_experiences

    def end_ep_callback(self):
        pass

    def reset(self):
        self.env.reset()
        self.state_info = None
        self._total_reward = 0
        self._state = None
        self.pending_reward_ticks.clear()
        self.action_history.clear()
        self.replay = defaultdict(lambda: defaultdict(list))
