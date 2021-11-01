import collections
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass
class BufferElement:
    state: np.ndarray
    action: int
    logp: float
    value: float
    reward: float
    terminal: bool


@dataclass
class MultiBufferElement:
    state: np.ndarray
    actions: np.ndarray
    logps: np.ndarray
    value: float
    reward: float
    terminal: bool


class Buffer:
    """Store a sequence of transitions, i.e., a trajectory.

    Args:
        size (int): Buffer capacity, i.e., the maximum number of stored transitions.
    """
    def __init__(self, size: int = 10000) -> None:
        self._pool: Deque[BufferElement] = collections.deque()
        self._size = size

    def put(self, state: np.ndarray, action: dict, reward: float, terminal: bool = False) -> None:
        self._pool.append(
            BufferElement(
                state=state.reshape(1, -1),
                action=action.get("action", 0),
                logp=action.get("logp", 0.0),
                value=action.get("value", 0.0),
                reward=reward,
                terminal=terminal
            )
        )
        if len(self._pool) > self._size:
            self._pool.popleft()
            # TODO: erase the older elements or raise MLE error?

    def get(self) -> dict:
        """Retrieve the latest trajectory segment."""
        if len(self._pool) == 0:
            return {}

        new_pool = collections.deque()
        if not self._pool[-1].terminal:
            new_pool.append(self._pool.pop())

        ret = {
            "states": np.concatenate([elem.state for elem in self._pool], axis=0),
            "actions": np.array([elem.action for elem in self._pool], dtype=np.int32),
            "logps": np.array([elem.logp for elem in self._pool], dtype=np.float32),
            "values": np.array([elem.value for elem in self._pool], dtype=np.float32),
            "rewards": np.array([elem.reward for elem in self._pool], dtype=np.float32),
            "last_value": self._pool[-1].value
        }

        self._pool = new_pool
        return ret


class MultiBuffer:
    """TODO
    """
    def __init__(self, agent_num: int, size: int = 10000) -> None:
        self._pool: Deque[MultiBufferElement] = collections.deque()
        self._agent_num = agent_num
        self._size = size

    def put(self, state: np.ndarray, action: dict, reward: float, terminal: bool = False) -> None:
        self._pool.append(
            MultiBufferElement(
                state=state.reshape(1, -1),
                actions=np.array(action.get("action", [0] * self._agent_num)).reshape(1, -1),
                logps=np.array(action.get("logp", [0.0] * self._agent_num)).reshape(1, -1),
                value=action.get("value", 0.0),
                reward=reward,
                terminal=terminal
            )
        )
        if len(self._pool) > self._size:
            self._pool.popleft()
            # TODO: erase the older elements or raise MLE error?

    def get(self) -> dict:
        """Retrieve the latest trajectory segment."""
        if len(self._pool) == 0:
            return {}

        new_pool = collections.deque()
        if not self._pool[-1].terminal:
            new_pool.append(self._pool.pop())

        ret = {
            "states": np.concatenate([elem.state for elem in self._pool], axis=0),  # [batch_size, state_dim]
            "actions": list(np.concatenate([elem.actions for elem in self._pool], axis=0).T),  # list of [batch_size]
            "logps": np.concatenate([elem.logps for elem in self._pool], axis=0),  # [batch_size, agent_num]
            "values": np.array([elem.value for elem in self._pool], dtype=np.float32),  # [batch_size]
            "rewards": np.array([elem.reward for elem in self._pool], dtype=np.float32),  # [batch_size]
            "terminals": np.array([elem.terminal for elem in self._pool], dtype=np.bool),  # [batch_size]
            "last_value": self._pool[-1].value  # Scalar
        }

        self._pool = new_pool
        return ret
