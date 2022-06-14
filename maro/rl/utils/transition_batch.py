# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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
    old_logps: np.ndarray = None  # 1D

    @property
    def size(self) -> int:
        return self.states.shape[0]

    def __post_init__(self) -> None:
        if SHAPE_CHECK_FLAG:
            assert len(self.states.shape) == 2 and self.states.shape[0] > 0
            assert len(self.actions.shape) == 2 and self.actions.shape[0] == self.states.shape[0]
            assert len(self.rewards.shape) == 1 and self.rewards.shape[0] == self.states.shape[0]
            assert self.next_states.shape == self.states.shape
            assert len(self.terminals.shape) == 1 and self.terminals.shape[0] == self.states.shape[0]

    def make_kth_sub_batch(self, i: int, k: int) -> TransitionBatch:
        return TransitionBatch(
            states=self.states[i::k],
            actions=self.actions[i::k],
            rewards=self.rewards[i::k],
            next_states=self.next_states[i::k],
            terminals=self.terminals[i::k],
            returns=self.returns[i::k] if self.returns is not None else None,
            advantages=self.advantages[i::k] if self.advantages is not None else None,
            old_logps=self.old_logps[i::k] if self.old_logps is not None else None,
        )

    def split(self, k: int) -> List[TransitionBatch]:
        return [self.make_kth_sub_batch(i, k) for i in range(k)]


@dataclass
class MultiTransitionBatch:
    states: np.ndarray  # 2D
    actions: List[np.ndarray]  # List of 2D
    rewards: List[np.ndarray]  # List of 1D
    next_states: np.ndarray  # 2D
    agent_states: List[np.ndarray]  # List of 2D
    next_agent_states: List[np.ndarray]  # List of 2D
    terminals: np.ndarray  # 1D

    returns: Optional[List[np.ndarray]] = None  # List of 1D
    advantages: Optional[List[np.ndarray]] = None  # List of 1D

    @property
    def size(self) -> int:
        return self.states.shape[0]

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

    def calc_returns(self, discount_factor: float) -> None:
        self.returns = [discount_cumsum(reward, discount_factor) for reward in self.rewards]

    def make_kth_sub_batch(self, i: int, k: int) -> MultiTransitionBatch:
        states = self.states[i::k]
        actions = [action[i::k] for action in self.actions]
        rewards = [reward[i::k] for reward in self.rewards]
        next_states = self.next_states[i::k]
        agent_states = [state[i::k] for state in self.agent_states]
        next_agent_states = [state[i::k] for state in self.next_agent_states]
        terminals = self.terminals[i::k]
        returns = None if self.returns is None else [r[i::k] for r in self.returns]
        advantages = None if self.advantages is None else [advantage[i::k] for advantage in self.advantages]
        return MultiTransitionBatch(
            states,
            actions,
            rewards,
            next_states,
            agent_states,
            next_agent_states,
            terminals,
            returns,
            advantages,
        )

    def split(self, k: int) -> List[MultiTransitionBatch]:
        return [self.make_kth_sub_batch(i, k) for i in range(k)]


def merge_transition_batches(batch_list: List[TransitionBatch]) -> TransitionBatch:
    return TransitionBatch(
        states=np.concatenate([batch.states for batch in batch_list], axis=0),
        actions=np.concatenate([batch.actions for batch in batch_list], axis=0),
        rewards=np.concatenate([batch.rewards for batch in batch_list], axis=0),
        next_states=np.concatenate([batch.next_states for batch in batch_list], axis=0),
        terminals=np.concatenate([batch.terminals for batch in batch_list]),
        returns=np.concatenate([batch.returns for batch in batch_list]),
        advantages=np.concatenate([batch.advantages for batch in batch_list]),
        old_logps=None
        if batch_list[0].old_logps is None
        else np.concatenate(
            [batch.old_logps for batch in batch_list],
        ),
    )
