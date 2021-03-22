# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import defaultdict
from multiprocessing import Pipe, Process
from typing import Callable

from maro.communication import Message, Proxy

from .message_enums import MsgTag, MsgKey


class AbsTrajectory(ABC):
    def __init__(self, env, record_path: bool = True):
        self.env = env
        self.step_index = None
        self.events = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.record_path = record_path

    def start(self, rollout_index: int = None):
        self.step_index = 0
        _, event, _ = self.env.step(None)
        state = self.get_state(event)
        if self.record_path:
            self.events.append(event)
            self.states.append(state)
        return state

    @property
    def path(self):
        return self.states, self.actions, self.rewards

    @abstractmethod
    def get_state(self, event) -> dict:
        pass

    @abstractmethod
    def get_action(self, action, event) -> dict:
        pass

    def get_reward(self) -> float:
        pass

    def step(self, action):
        assert self.events, "start() must be called first."
        self.step_index += 1
        env_action = self.get_action(action, self.events[-1])
        if len(env_action) == 1:
            env_action = list(env_action.values())[0]
        _, event, done = self.env.step(env_action)
        if self.record_path:
            self.actions.append(action)
            self.rewards.append(self.get_reward())
        assert len(self.events) == len(self.states) == len(self.actions) == len(self.rewards)
        if not done:
            state = self.get_state(event)
            if self.record_path:
                self.events.append(event)
                self.states.append(state)
            return state

    def on_env_feedback(self):
        pass

    def on_finish(self):
        pass

    def reset(self):
        self.env.reset()
        self.events.clear()
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def flush(self):
        self.events = self.events[-1:] if len(self.events) == len(self.actions) + 1 else []
        self.states = self.states[-1:] if len(self.states) == len(self.actions) + 1 else []
        self.actions.clear()
        self.rewards.clear()
