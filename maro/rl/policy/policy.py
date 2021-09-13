# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import List

from maro.communication import Proxy


class AbsPolicy(ABC):
    """Abstract policy class.

    Args:
        name (str): Unique identifier for the policy.

    """
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    @property
    def name(self):
        return self._name

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
    """Policy that learns from simulation experiences.

    Reinforcement learning (RL) policies should inherit from this.

    Args:
        name (str): Name of the policy.
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.exploration_params = {}
        self.greedy = True

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError

    def record(self, key: str, state, action, reward, next_state, terminal: bool):
        pass

    def get_rollout_info(self):
        pass

    def get_batch_loss(self, batch: dict, explicit_grad: bool = False):
        pass

    def data_parallel(self, *args, **kwargs):
        self.remote = True
        self._proxy = Proxy(*args, **kwargs)

    def data_parallel_with_existing_proxy(self, proxy):
        self.remote = True
        self._proxy = proxy

    def exit_data_parallel(self):
        self.remote = False
        if hasattr(self, '_proxy'):
            self._proxy.close()

    def learn_with_data_parallel(self):
        pass

    def update(self, loss_info_list: List[dict]):
        pass

    def learn(self, batch: dict):
        """Perform policy improvement based on a single data batch collected from one or more roll-out instances."""
        pass

    def improve(self):
        pass

    def exploit(self):
        self.greedy = True

    def explore(self):
        self.greedy = False

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
