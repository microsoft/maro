from dataclasses import dataclass
from typing import List

import numpy as np

from .objects import SHAPE_CHECK_FLAG


@dataclass
class TransitionBatch:
    policy_name: str
    states: np.ndarray  # 2D
    actions: np.ndarray  # 2D
    rewards: np.ndarray  # 1D
    next_states: np.ndarray  # 2D
    terminals: np.ndarray  # 1D

    def __post_init__(self) -> None:
        if SHAPE_CHECK_FLAG:
            assert len(self.states.shape) == 2 and self.states.shape[0] > 0
            assert len(self.actions.shape) == 2 and self.actions.shape[0] == self.states.shape[0]
            assert len(self.rewards.shape) == 1 and self.rewards.shape[0] == self.states.shape[0]
            assert self.next_states.shape == self.states.shape
            assert len(self.terminals.shape) == 1 and self.terminals.shape[0] == self.states.shape[0]


@dataclass
class MultiTransitionBatch:
    policy_names: List[str]
    states: np.ndarray  # 2D
    actions: List[np.ndarray]  # 2D
    rewards: List[np.ndarray]  # 1D
    next_states: np.ndarray  # 2D
    policy_states: List[np.ndarray]  # 2D
    next_policy_states: List[np.ndarray]  # 2D
    terminals: np.ndarray  # 1D

    def __post_init__(self) -> None:
        if SHAPE_CHECK_FLAG:
            assert len(self.states.shape) == 2 and self.states.shape[0] > 0

            assert len(self.actions) == len(self.rewards)
            assert len(self.policy_states) == len(self.actions)
            for i in range(len(self.actions)):
                assert len(self.actions[i].shape) == 2 and self.actions[i].shape[0] == self.states.shape[0]
                assert len(self.rewards[i].shape) == 1 and self.rewards[i].shape[0] == self.states.shape[0]
                assert len(self.policy_states[i].shape) == 2
                assert self.policy_states[i].shape[0] == self.states.shape[0]

            assert len(self.terminals.shape) == 1 and self.terminals.shape[0] == self.states.shape[0]
            assert self.next_states.shape == self.states.shape

            assert len(self.next_policy_states) == len(self.policy_states)
            for i in range(len(self.next_policy_states)):
                assert self.policy_states[i].shape == self.next_policy_states[i].shape
