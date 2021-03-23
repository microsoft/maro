# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import defaultdict
from multiprocessing import Pipe, Process
from typing import Callable

from maro.communication import Message, Proxy
from maro.simulator import Env

from .message_enums import MsgTag, MsgKey

MAX_LOSS = 1e8


class AbsEnvWrapper(ABC):
    """Environment wrapper that performs various shaping and other roll-out related logic.

    Args:
        env (Env): Environment instance.
        record_path (bool): If True, the steps during roll-out will be recorded sequentially. This
            includes states, actions and rewards. The decision events themselves will also be recorded
            for hindsight reward evaluation purposes. Defaults to True. 
        hindsight_reward_window (int): Number of ticks required after a decision event to evaluate
            the reward for the action taken for that event. Defaults to 0, which rewards are evaluated immediately
            after executing an action.
    """
    def __init__(self, env: Env, record_path: bool = True, hindsight_reward_window: int = 0):
        self.env = env
        self.step_index = None
        self.replay_memory = defaultdict(lambda: {key: [] for key in ["S", "A", "R", "S_", "loss"]})
        self.events = []
        self.acting_agents = []
        self.record_path = record_path
        self.hindsight_reward_window = hindsight_reward_window
        self._pending_reward_idx = 0

    def start(self, rollout_index: int = None):
        self.step_index = 0
        self._pending_reward_idx = 0
        _, event, _ = self.env.step(None)
        return self._on_new_event(event)

    @property
    def replay(self):
        return {
            agent_id: {k: vals[:len(replay["R"])] for k, vals in replay.items()}
            for agent_id, replay in self.replay_memory.items()
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
        
        This can be left blank if rewards are evaluated in hindsight.
        """
        pass

    def get_hindsight_reward(self, event):
        """Get the reward for an action that occurred a certain number of ticks ago.

        If implemented, whatever value ``get_reward`` gives will be ignored in the output of ``get_path``.
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
            if self.hindsight_reward_window:
                for agent_id, action in action_by_agent.items():
                    self.replay_memory[agent_id]["A"].append(action)
                self._assign_hindsight_rewards(tick=event.tick if not done else None)
            else:
                reward = self.get_reward()
                for agent_id, action in action_by_agent.items():
                    self.replay_memory[agent_id]["A"].append(action)
                    self.replay_memory[agent_id]["R"].append(reward)
                self._pending_reward_idx += 1

        if not done:
            return self._on_new_event(event)

    def reset(self):
        self.env.reset()
        self.events.clear()
        self.replay_memory = defaultdict(lambda: {key: [] for key in ["S", "A", "R", "S_", "loss"]})

    def flush(self):
        for agent_id in self.replay_memory:
            num_complete = len(self.replay_memory[agent_id]["R"])
            del self.replay_memory[agent_id]["S"][:num_complete]
            del self.replay_memory[agent_id]["A"][:num_complete]
            del self.replay_memory[agent_id]["R"][:num_complete]
            del self.replay_memory[agent_id]["S_"][:num_complete]
            del self.replay_memory[agent_id]["loss"][:num_complete]

        del self.events[:self._pending_reward_idx]
        self._pending_reward_idx = 0

    def _on_new_event(self, event):
        state_by_agent = self.get_state(event)
        if self.record_path:
            self.events.append((event, list(state_by_agent.keys())))
            for agent_id, state in state_by_agent.items():
                if self.replay_memory[agent_id]["S"]:
                    self.replay_memory[agent_id]["S_"].append(state)
                self.replay_memory[agent_id]["loss"].append(MAX_LOSS)
                self.replay_memory[agent_id]["S"].append(state)
       
            # for agent_id, exp in self.replay_memory.items():
            #     ns, na, nr, ns_ = len(exp["S"]), len(exp["A"]), len(exp["R"]), len(exp["S_"])
            #     print(f"agent_id: {agent_id}, state: {ns}, action: {na}, reward: {nr}, state_: {ns_}")

        return state_by_agent

    def _assign_hindsight_rewards(self, tick=None):
        while (
            self._pending_reward_idx < len(self.events) and
            (tick is None or tick - self.events[self._pending_reward_idx][0].tick >= self.hindsight_reward_window)
        ):  
            event, acting_agents = self.events[self._pending_reward_idx]
            hindsight_reward = self.get_hindsight_reward(event) 
            for agent_id in acting_agents:
                self.replay_memory[agent_id]["R"].append(hindsight_reward)
            self._pending_reward_idx += 1
