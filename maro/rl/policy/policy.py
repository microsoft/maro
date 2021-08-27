# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class AbsPolicy(ABC):
    """Abstract policy class.

    Args:
        name (str): Unique identifier for the policy.

    """
    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self.agents = set()

    @property
    def name(self):
        return self._name

    def add_agent(self, agent: str):
        self.agents.add(agent)

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError


class NullPolicy(AbsPolicy):
    """Dummy policy that does nothing.

    Note that the meaning of a "None" action may depend on the scenario.
    """
    def choose_action(self, state):
        return None


class RLPolicy(AbsPolicy):
    class Buffer:
        """Sequence of transitions for an agent.

        Args:
            states: Sequence of ``State`` objects traversed during simulation.
            actions: Sequence of actions taken in response to the states.
            rewards: Sequence of rewards received as a result of the actions.
            info: Sequence of each transition's auxillary information.
        """
        def __init__(self, state_dim: int, action_dim: int = 1, max_len: int = 10000):
            self.states = np.zeros((max_len, state_dim), dtype=np.float32)
            if action_dim == 1:
                self.actions = np.zeros(max_len, dtype=np.float32)
            else:
                self.actions = np.zeros((max_len, action_dim), dtype=np.float32)
            self.rewards = np.zeros(max_len, dtype=np.float32)
            self.terminal = np.zeros(max_len, dtype=np.bool)
            self.max_len = max_len

            self._ptr = 0
            self._last_ptr = 0

        @abstractmethod
        def put(self, transition):
            raise NotImplementedError

        @abstractmethod
        def get(self):
            raise NotImplementedError

    """Policy that learns from simulation experiences.

    Reinforcement learning (RL) policies should inherit from this.

    Args:
        name (str): Name of the policy.
    """
    def __init__(self, name: str, remote: bool = False):
        super().__init__(name)
        self.remote = remote

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError

    def record(self, key: str, state, action, reward, next_state, terminal: bool):
        pass
    
    @abstractmethod
    def get_rollout_info(self):
        raise NotImplementedError

    @abstractmethod
    def get_batch_loss(self, batch: dict, explicit_grad: bool = False):
        raise NotImplementedError

    @abstractmethod
    def update_with_multi_loss_info(self, loss_info_list: List[dict]):
        pass

    @abstractmethod
    def learn_from_multi_trajectories(self, trajectories: List[dict]):
        """Perform policy improvement based on a list of trajectories obtained from parallel rollouts."""
        raise NotImplementedError

    def exploit(self):
        pass

    def explore(self):
        pass

    def exploration_step(self):
        pass

    @property
    def exploration_params(self):
        return None

    @abstractmethod
    def get_state(self):
        """Return the current state of the policy.

        The implementation must be in correspondence with that of ``set_state``. For example, if a torch model
        is contained in the policy, ``get_state`` may include a call to ``state_dict()`` on the model, while
        ``set_state`` should accordingly include ``load_state_dict()``.
        """
        pass

    @abstractmethod
    def set_state(self, policy_state):
        """Set the policy state to ``policy_state``.

        The implementation must be in correspondence with that of ``get_state``. For example, if a torch model
        is contained in the policy, ``set_state`` may include a call to ``load_state_dict()`` on the model, while
        ``get_state`` should accordingly include ``state_dict()``.
        """
        pass

    def load(self, path: str):
        """Load the policy state from disk."""
        pass

    def save(self, path: str):
        """Save the policy state to disk."""
        pass
