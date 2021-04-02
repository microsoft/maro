# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from abc import ABC, abstractmethod
from collections import defaultdict

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
        self.env = env
        self.step_index = None
        self.replay = defaultdict(lambda: defaultdict(list))
        self.state_info = None  # context for converting model output to actions that can be executed by the env
        self.save_replay = save_replay
        self.reward_eval_delay = reward_eval_delay
        self._pending_reward_idx = 0
        self._event_ticks = []  # for delayed reward evaluation
        self._action_history = []  # for delayed reward evaluation
        self._tot_raw_step_time = 0
        self._tot_step_time = 0

    def start(self, rollout_index: int = None):
        self.step_index = 0
        self._pending_reward_idx = 0
        _, event, _ = self.env.step(None)
        state_by_agent = self.get_state(event)
        if self.save_replay:
            for agent_id, state in state_by_agent.items():
                replay = self.replay[agent_id]
                if replay["S"]:
                    replay["S_"].append(state)
                replay["S"].append(state)
                assert len(replay["S_"]) == len(replay["A"]) == len(replay["S"]) - 1

        return state_by_agent

    @property
    def replay_memory(self):
        return {
            agent_id: {k: vals[:len(replay["R"])] for k, vals in replay.items()}
            for agent_id, replay in self.replay.items()
        }

    @property
    def metrics(self):
        return self.env.metrics

    @abstractmethod
    def get_state(self, event) -> dict:
        pass

    @abstractmethod
    def get_action(self, action) -> dict:
        pass

    @abstractmethod
    def get_reward(self, tick: int = None) -> float:
        """User-defined reward evaluation.

        Args:
            tick (int): If given, the action that occured at this tick will be evaluated (useful for delayed reward
            evaluation). Otherwise, the reward is evaluated for the latest action. Defaults to None.

        """
        pass

    def step(self, action_by_agent: dict):
        t0 = time.time()
        self.step_index += 1
        self._event_ticks.append(self.env.tick)
        env_action = self.get_action(action_by_agent)
        self._action_history.append(env_action)
        if len(env_action) == 1:
            env_action = list(env_action.values())[0]
        t1 = time.time()
        _, event, done = self.env.step(None)
        t2 = time.time()
        self._tot_raw_step_time += t2 - t1

        if self.save_replay:
            if self.reward_eval_delay:
                for agent_id, action in action_by_agent.items():
                    if isinstance(action, tuple):
                        self.replay[agent_id]["A"].append(action[0])
                        self.replay[agent_id]["LOGP"].append(action[1])
                    else:
                        self.replay[agent_id]["A"].append(action)
                """
                If roll-out is complete, evaluate rewards for all remaining events except the last.
                Otherwise, evaluate rewards only for events at least self.reward_eval_delay ticks ago.
                """
                for i, (tick, action) in enumerate(
                    zip(self._event_ticks[self._pending_reward_idx:], self._action_history[self._pending_reward_idx:])
                ):
                    if not done and self.env.tick - tick < self.reward_eval_delay:
                        self._pending_reward_idx += i
                        break
                    reward_dict = self.get_reward(tick=tick)
                    for agent_id in action:
                        if len(self.replay[agent_id]["R"]) < len(self.replay[agent_id]["S_"]):
                            self.replay[agent_id]["R"].append(reward_dict[agent_id])

                if done:
                    self._pending_reward_idx = len(self._event_ticks) - 1
            else:
                reward_dict = self.get_reward()
                for agent_id, action in action_by_agent.items():
                    if isinstance(action, tuple):
                        self.replay[agent_id]["A"].append(action[0])
                        self.replay[agent_id]["LOGP"].append(action[1])
                    else:
                        self.replay[agent_id]["A"].append(action)
                    if len(self.replay[agent_id]["R"]) < len(self.replay[agent_id]["S_"]):
                        self.replay[agent_id]["R"].append(reward_dict[agent_id])
                    self._pending_reward_idx += 1

        if not done:
            state_by_agent = self.get_state(event)
            if self.save_replay:
                for agent_id, state in state_by_agent.items():
                    replay = self.replay[agent_id]
                    if replay["S"]:
                        replay["S_"].append(state)
                    replay["S"].append(state)
                    assert len(replay["S_"]) == len(replay["A"]) == len(replay["S"]) - 1

            t3 = time.time()
            self._tot_step_time += t3 - t0
            return state_by_agent

        print(f"total raw step time: {self._tot_raw_step_time}")
        print(f"total step time: {self._tot_step_time}")
        self._tot_raw_step_time = 0
        self._tot_step_time = 0

    def reset(self):
        self.env.reset()
        self.state_info = None
        self._event_ticks.clear()
        self._action_history.clear()
        self.replay = defaultdict(lambda: defaultdict(list))

    def flush(self):
        for replay in self.replay.values():
            num_complete = len(replay["R"])
            for vals in replay.values():
                del vals[:num_complete]

        del self._event_ticks[:self._pending_reward_idx]
        del self._action_history[:self._pending_reward_idx]
        self._pending_reward_idx = 0
