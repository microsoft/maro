# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import numpy as np
import torch

from maro.rl.utils import SHAPE_CHECK_FLAG, match_shape, ndarray_to_tensor


class AbsPolicy(object, metaclass=ABCMeta):
    """
    Policy. A policy takes states as inputs and generates actions as outputs. A policy cannot update itself. It has to
    be updated by external trainers through public interfaces.
    """

    def __init__(self, name: str, trainable: bool) -> None:
        """
        Args:
            name (str): Name of this policy.
            trainable (bool): Whether this policy is trainable.
        """
        super(AbsPolicy, self).__init__()
        self._name = name
        self._trainable = trainable

    @abstractmethod
    def get_actions(self, states: object) -> object:
        """
        Get actions according to states.

        Args:
            states (object): States.

        Returns:
            Actions.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @property
    def trainable(self) -> bool:
        return self._trainable

    def set_name(self, name: str) -> None:
        self._name = name


class DummyPolicy(AbsPolicy):
    """
    Dummy policy that takes no actions.
    """
    def __init__(self) -> None:
        super(DummyPolicy, self).__init__(name='DUMMY_POLICY', trainable=False)

    def get_actions(self, states: object) -> object:
        return None


class RuleBasedPolicy(AbsPolicy, metaclass=ABCMeta):
    """
    Rule-based policy. The user should implement the rule of this policy, and a rule-based policy is not trainable.
    """
    def __init__(self, name: str) -> None:
        super(RuleBasedPolicy, self).__init__(name=name, trainable=False)

    def get_actions(self, states: object) -> object:
        return self._rule(states)

    @abstractmethod
    def _rule(self, states: object) -> object:
        raise NotImplementedError


class RLPolicy(AbsPolicy, metaclass=ABCMeta):
    """
    Reinforcement learning policy.
    """
    def __init__(
        self,
        name: str,
        state_dim: int,
        action_dim: int,
        trainable: bool = True
    ) -> None:
        """
        Args:
            name (str): Name of the policy.
            state_dim (int): Dimension of states.
            action_dim (int): Dimension of actions.
            trainable (bool): Whether this policy is trainable. Defaults to True.
        """
        super(RLPolicy, self).__init__(name=name, trainable=trainable)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._is_exploring = False

        self._device: Optional[torch.device] = None

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def is_exploring(self) -> bool:
        """
        Whether this policy is under exploring mode.
        """
        return self._is_exploring

    def explore(self) -> None:
        """
        Set the policy to exploring mode.
        """
        self._is_exploring = True

    def exploit(self) -> None:
        """
        Set the policy to exploiting mode.
        """
        self._is_exploring = False

    @abstractmethod
    def step(self, loss: torch.Tensor) -> None:
        """
        Run a training step to update the policy.

        Args:
            loss (torch.Tensor): Loss used to update the policy.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get the gradients with respect to all parameters of the internal nets according to the given loss.

        Args:
            loss (torch.tensor): Loss used to update the model.

        Returns:
            A dict that contains gradients of the internal nets for all parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_gradients(self, grad: dict) -> None:
        raise NotImplementedError

    def get_actions(self, states: np.ndarray) -> np.ndarray:
        return self.get_actions_tensor(ndarray_to_tensor(states, self._device)).cpu().numpy()

    def get_actions_tensor(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get actions according to states. Takes torch.Tensor as inputs and returns torch.Tensor.

        Args:
            states (torch.Tensor): States.

        Returns:
            Actions, a torch.Tensor.
        """
        assert self._shape_check(states=states), \
            f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."
        actions = self._get_actions_impl(states, self._is_exploring)
        assert self._shape_check(states=states, actions=actions), \
            f"Actions shape check failed. Expecting: {(states.shape[0], self.action_dim)}, actual: {actions.shape}."
        if SHAPE_CHECK_FLAG:
            assert self._post_check(states=states, actions=actions)
        return actions

    @abstractmethod
    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def freeze(self) -> None:
        """
        (Partially) freeze the current model. The users should write their own strategy to determine the list of
        parameters to freeze.
        """
        raise NotImplementedError

    @abstractmethod
    def unfreeze(self) -> None:
        """
        (Partially) unfreeze the current model. The users should write their own strategy to determine the list of
        parameters to unfreeze.
        """
        raise NotImplementedError

    @abstractmethod
    def eval(self) -> None:
        """
        Switch the policy to evaluating mode.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        """
        Switch the policy to training mode.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> object:
        """
        Get the state of the policy.
        """
        raise NotImplementedError

    @abstractmethod
    def set_state(self, policy_state: object) -> None:
        """
        Set the state of the policy.
        """
        raise NotImplementedError

    @abstractmethod
    def soft_update(self, other_policy: RLPolicy, tau: float) -> None:
        """
        Soft update the policy's parameters according to another policy.

        Args:
            other_policy (AbsNet): The source policy. Must has same type with the current policy.
            tau (float): Soft update coefficient.
        """
        raise NotImplementedError

    def _shape_check(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> bool:
        if not SHAPE_CHECK_FLAG:
            return True
        else:
            if states.shape[0] == 0:
                return False
            if not match_shape(states, (None, self.state_dim)):
                return False

            if actions is not None:
                if not match_shape(actions, (states.shape[0], self.action_dim)):
                    return False
            return True

    @abstractmethod
    def _post_check(self, states: torch.Tensor, actions: torch.Tensor) -> bool:
        raise NotImplementedError

    def to_device(self, device: torch.device) -> None:
        if self._device is None:
            self._device = device
            self._to_device_impl(device)
            print(f"Assign policy {self.name} to device {device}")
        elif self._device == device:
            print(f"Policy {self.name} has already been assigned to {device}. No need to take further actions.")
        else:
            raise ValueError(
                f"Policy {self.name} has already been assigned to device {self._device} "
                f"and cannot be re-assigned to device {device}"
            )

    @abstractmethod
    def _to_device_impl(self, device: torch.device) -> None:
        pass


if __name__ == '__main__':
    data = [AbsPolicy('Jack', True), AbsPolicy('Tom', True), DummyPolicy(), DummyPolicy()]
    for policy in data:
        print(policy.name)
