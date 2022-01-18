from dataclasses import dataclass
from typing import List

import numpy as np

from . import discount_cumsum
from .objects import SHAPE_CHECK_FLAG


@dataclass
class TransitionBatch:
    states: np.ndarray  # 2D
    actions: np.ndarray  # 2D
    rewards: np.ndarray  # 1D
    next_states: np.ndarray  # 2D
    terminals: np.ndarray  # 1D
    returns: np.ndarray = None  # 1D
    advantages: np.ndarray = None  # 1D

    def __post_init__(self) -> None:
        if SHAPE_CHECK_FLAG:
            assert len(self.states.shape) == 2 and self.states.shape[0] > 0
            assert len(self.actions.shape) == 2 and self.actions.shape[0] == self.states.shape[0]
            assert len(self.rewards.shape) == 1 and self.rewards.shape[0] == self.states.shape[0]
            assert self.next_states.shape == self.states.shape
            assert len(self.terminals.shape) == 1 and self.terminals.shape[0] == self.states.shape[0]

    def calc_returns(self, discount_factor: float) -> None:
        self.returns = discount_cumsum(self.rewards, discount_factor)

    def split(self, k: int):
        return [
            TransitionBatch(
                self.states[i::k],
                self.actions[i::k],
                self.rewards[i::k],
                self.next_states[i::k],
                self.terminals[i::k],
                returns=self.returns[i::k] if self.returns is not None else None,
                advantages=self.advantages[i::k] if self.advantages is not None else None
            )
            for i in range(k)
        ]


@dataclass
class MultiTransitionBatch:
    states: np.ndarray  # 2D
    actions: List[np.ndarray]  # 2D
    rewards: List[np.ndarray]  # 1D
    next_states: np.ndarray  # 2D
    agent_states: List[np.ndarray]  # 2D
    next_agent_states: List[np.ndarray]  # 2D
    terminals: np.ndarray  # 1D

    def __post_init__(self) -> None:
        if SHAPE_CHECK_FLAG:
            assert len(self.states.shape) == 2 and self.states.shape[0] > 0

            assert len(self.actions) == len(self.rewards)
            assert len(self.agent_states) == len(self.actions)
            for i in range(len(self.actions)):
                assert len(self.actions[i].shape) == 2 and self.actions[i].shape[0] == self.states.shape[0]
                assert len(self.rewards[i].shape) == 1 and self.rewards[i].shape[0] == self.states.shape[0]
                assert len(self.agent_states[i].shape) == 2
                assert self.agent_states[i].shape[0] == self.states.shape[0]

            assert len(self.terminals.shape) == 1 and self.terminals.shape[0] == self.states.shape[0]
            assert self.next_states.shape == self.states.shape

            assert len(self.next_agent_states) == len(self.agent_states)
            for i in range(len(self.next_agent_states)):
                assert self.agent_states[i].shape == self.next_agent_states[i].shape
