# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable

from maro.simulator import Env


class AbsEnvWrapper(ABC):
    """Environment wrapper that performs various shaping and other roll-out related logic.

    Args:
        env (Env): Environment instance.
        record_path (bool): If True, the steps during roll-out will be recorded sequentially. This
            includes states, actions and rewards. The decision events themselves will also be recorded
            for delayed reward evaluation purposes. Defaults to True. 
        reward_eval_delay (int): Number of ticks required after a decision event to evaluate the reward
            for the action taken for that event. Defaults to 0, which rewards are evaluated immediately
            after executing an action.
    """
    def __init__(self, env: Env, record_path: bool = True, reward_eval_delay: int = 0):
        self.env = env
        self.step_index = None
        self.replay = defaultdict(lambda: defaultdict(list))
        self.events = []
        self.acting_agents = []
        self.record_path = record_path
        self.reward_eval_delay = reward_eval_delay
        self._pending_reward_idx = 0

    def start(self, rollout_index: int = None):
        self.step_index = 0
        self._pending_reward_idx = 0
        _, event, _ = self.env.step(None)
        state_by_agent = self.get_state(event)
        if self.record_path:
            self.events.append((event, list(state_by_agent.keys())))
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
    def get_action(self, action, event) -> dict:
        pass

    def get_reward(self) -> float:
        """Get the immediate reward for an action.
        
        This can be left blank if ``get_reward_for`` is implemented.
        """
        pass

    def get_reward_for(self, event) -> float:
        """Get the reward for an action in response to an event that occurred a certain number of ticks ago.

        If implemented, whatever value ``get_reward`` gives will be ignored in the output of ``replay_memory``.
        If left blank, ``get_reward`` must be implemented.
        """
        pass

    def step(self, action_by_agent: dict):
        assert self.events, "start() must be called first."
        self.step_index += 1
        env_action = self.get_action(action_by_agent, self.events[-1][0])
        if len(env_action) == 1:
            env_action = list(env_action.values())[0]
        _, event, done = self.env.step(env_action)

        if self.record_path:
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
                for i, (evt, agents) in enumerate(self.events[self._pending_reward_idx:]):
                    if not done and event.tick - evt.tick < self.reward_eval_delay:
                        self._pending_reward_idx += i                        
                        break
                    reward = self.get_reward_for(evt)
                    for agent_id in agents:
                        if len(self.replay[agent_id]["R"]) < len(self.replay[agent_id]["S_"]):
                            self.replay[agent_id]["R"].append(reward)

                if done:
                    self._pending_reward_idx = len(self.events) - 1
            else:
                for agent_id, action in action_by_agent.items():
                    if isinstance(action, tuple):
                        self.replay[agent_id]["A"].append(action[0])
                        self.replay[agent_id]["LOGP"].append(action[1])
                    else:
                        self.replay[agent_id]["A"].append(action)
                    self.replay[agent_id]["R"].append(reward)
                    self._pending_reward_idx += 1

        if not done:
            state_by_agent = self.get_state(event)
            if self.record_path:
                self.events.append((event, list(state_by_agent.keys())))
                for agent_id, state in state_by_agent.items():
                    replay = self.replay[agent_id]
                    if replay["S"]:
                        replay["S_"].append(state)
                    replay["S"].append(state)
                    assert len(replay["S_"]) == len(replay["A"]) == len(replay["S"]) - 1 
        
            return state_by_agent

    def reset(self):
        self.env.reset()
        self.events.clear()
        self.replay = defaultdict(lambda: defaultdict(list))

    def flush(self):
        for replay in self.replay.values():
            num_complete = len(replay["R"])
            for vals in replay.values():
                del vals[:num_complete]

        del self.events[:self._pending_reward_idx]
        self._pending_reward_idx = 0
