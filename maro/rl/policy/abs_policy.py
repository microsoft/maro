# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from maro.rl.utils import SHAPE_CHECK_FLAG, match_shape, ndarray_to_tensor


class AbsPolicy(object, metaclass=ABCMeta):
    """Abstract policy class. A policy takes states as inputs and generates actions as outputs. A policy cannot
        update itself. It has to be updated by external trainers through public interfaces.

    Args:
        name (str): Name of this policy.
        trainable (bool): Whether this policy is trainable.
    """

    def __init__(self, name: str, trainable: bool) -> None:
        super(AbsPolicy, self).__init__()
        self._name = name
        self._trainable = trainable

    @abstractmethod
    def get_actions(self, states: Union[list, np.ndarray]) -> Any:
        """Get actions according to states.

        Args:
            states (Union[list, np.ndarray]): States.

        Returns:
            actions (Any): Actions.
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

    @abstractmethod
    def explore(self) -> None:
        """Set the policy to exploring mode."""
        raise NotImplementedError

    @abstractmethod
    def exploit(self) -> None:
        """Set the policy to exploiting mode."""
        raise NotImplementedError

    @abstractmethod
    def eval(self) -> None:
        """Switch the policy to evaluation mode."""
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        """Switch the policy to training mode."""
        raise NotImplementedError

    def to_device(self, device: torch.device) -> None:
        pass


class DummyPolicy(AbsPolicy):
    """Dummy policy that takes no actions."""

    def __init__(self) -> None:
        super(DummyPolicy, self).__init__(name="DUMMY_POLICY", trainable=False)

    def get_actions(self, states: Union[list, np.ndarray]) -> None:
        return None

    def explore(self) -> None:
        pass

    def exploit(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass


class RuleBasedPolicy(AbsPolicy, metaclass=ABCMeta):
    """Rule-based policy. The user should define the rule of this policy, and a rule-based policy is not trainable."""

    def __init__(self, name: str) -> None:
        super(RuleBasedPolicy, self).__init__(name=name, trainable=False)

    def get_actions(self, states: list) -> list:
        return self._rule(states)

    @abstractmethod
    def _rule(self, states: list) -> list:
        raise NotImplementedError

    def explore(self) -> None:
        pass

    def exploit(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass


class RLPolicy(AbsPolicy, metaclass=ABCMeta):
    """Reinforcement learning policy.

    Args:
        name (str): Name of the policy.
        state_dim (int): Dimension of states.
        action_dim (int): Dimension of actions.
        trainable (bool, default=True): Whether this policy is trainable.
    """

    def __init__(
        self,
        name: str,
        state_dim: int,
        action_dim: int,
        is_discrete_action: bool,
        trainable: bool = True,
    ) -> None:
        super(RLPolicy, self).__init__(name=name, trainable=trainable)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._is_exploring = False

        self._device: Optional[torch.device] = None

        self.is_discrete_action = is_discrete_action

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def is_exploring(self) -> bool:
        """Whether this policy is under exploring mode."""
        return self._is_exploring

    def explore(self) -> None:
        """Set the policy to exploring mode."""
        self._is_exploring = True

    def exploit(self) -> None:
        """Set the policy to exploiting mode."""
        self._is_exploring = False

    @abstractmethod
    def train_step(self, loss: torch.Tensor) -> None:
        """Run a training step to update the policy according to the given loss.

        Args:
            loss (torch.Tensor): Loss used to update the policy.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get the gradients with respect to all parameters of the internal nets according to the given loss.

        Args:
            loss (torch.tensor): Loss used to update the model.

        Returns:
            grad (Dict[str, torch.Tensor]): A dict that contains gradients of the internal nets for all parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_gradients(self, grad: dict) -> None:
        """Apply gradients to the net to update all parameters.

        Args:
            grad (Dict[str, torch.Tensor]): A dict that contains gradients for all parameters.
        """
        raise NotImplementedError

    def get_actions(self, states: np.ndarray) -> np.ndarray:
        actions = self.get_actions_tensor(ndarray_to_tensor(states, device=self._device))
        return actions.detach().cpu().numpy()

    def get_actions_tensor(self, states: torch.Tensor) -> torch.Tensor:
        assert self._shape_check(
            states=states,
        ), f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."

        actions = self._get_actions_impl(states)

        assert self._shape_check(
            states=states,
            actions=actions,
        ), f"Actions shape check failed. Expecting: {(states.shape[0], self.action_dim)}, actual: {actions.shape}."

        return actions

    def get_actions_with_probs(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._shape_check(
            states=states,
        ), f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."

        actions, probs = self._get_actions_with_probs_impl(states)

        assert self._shape_check(
            states=states,
            actions=actions,
        ), f"Actions shape check failed. Expecting: {(states.shape[0], self.action_dim)}, actual: {actions.shape}."
        assert len(probs.shape) == 1 and probs.shape[0] == states.shape[0]

        return actions, probs

    def get_actions_with_logps(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._shape_check(
            states=states,
        ), f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."

        actions, logps = self._get_actions_with_logps_impl(states)

        assert self._shape_check(
            states=states,
            actions=actions,
        ), f"Actions shape check failed. Expecting: {(states.shape[0], self.action_dim)}, actual: {actions.shape}."
        assert len(logps.shape) == 1 and logps.shape[0] == states.shape[0]

        return actions, logps

    def get_states_actions_probs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        assert self._shape_check(
            states=states,
        ), f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."

        probs = self._get_states_actions_probs_impl(states, actions)

        assert len(probs.shape) == 1 and probs.shape[0] == states.shape[0]

        return probs

    def get_states_actions_logps(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        assert self._shape_check(
            states=states,
        ), f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."

        logps = self._get_states_actions_logps_impl(states, actions)

        assert len(logps.shape) == 1 and logps.shape[0] == states.shape[0]

        return logps

    @abstractmethod
    def _get_actions_impl(self, states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_actions_with_probs_impl(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _get_actions_with_logps_impl(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _get_states_actions_probs_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def freeze(self) -> None:
        """(Partially) freeze the current model. The users should write their own strategy to determine which
        parameters to freeze.
        """
        raise NotImplementedError

    @abstractmethod
    def unfreeze(self) -> None:
        """(Partially) unfreeze the current model. The users should write their own strategy to determine which
        parameters to freeze.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> dict:
        """Get the state of the policy."""
        raise NotImplementedError

    @abstractmethod
    def set_state(self, policy_state: dict) -> None:
        """Set the state of the policy."""
        raise NotImplementedError

    @abstractmethod
    def soft_update(self, other_policy: RLPolicy, tau: float) -> None:
        """Soft update the policy's parameters according to another policy.

        Args:
            other_policy (AbsNet): The source policy. Must has same type with the current policy.
            tau (float): Soft update coefficient.
        """
        raise NotImplementedError

    def _shape_check(
        self,
        states: torch.Tensor,
        actions: torch.Tensor = None,
    ) -> bool:
        """Check whether the states and actions have valid shapes.

        Args:
            states (torch.Tensor): State tensor.
            actions (torch.Tensor, default=None): Action tensor. If it is None, it means we only check state tensor's
                shape.

        Returns:
            valid_flag (bool): whether the states and actions have valid shapes.
        """
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
        """Check whether the generated action tensor is valid, i.e., has matching shape with states tensor.

        Args:
            states (torch.Tensor): State tensor.
            actions (torch.Tensor): Action tensor.

        Returns:
            valid_flag (bool): whether the action tensor is valid.
        """
        raise NotImplementedError

    def to_device(self, device: torch.device) -> None:
        """Assign the current policy to a specific device.

        Args:
            device (torch.device): The target device.
        """
        if self._device is None:
            self._device = device
            self._to_device_impl(device)
        elif self._device != device:
            raise ValueError(
                f"Policy {self.name} has already been assigned to device {self._device} "
                f"and cannot be re-assigned to device {device}",
            )

    @abstractmethod
    def _to_device_impl(self, device: torch.device) -> None:
        """Implementation of `to_device`."""
        raise NotImplementedError
