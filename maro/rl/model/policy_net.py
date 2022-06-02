# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch.nn
from torch.distributions import Categorical

from maro.rl.utils import SHAPE_CHECK_FLAG, match_shape

from .abs_net import AbsNet


class PolicyNet(AbsNet, metaclass=ABCMeta):
    """Base class for all nets that serve as policy cores. It has the concept of 'state' and 'action'.

    Args:
        state_dim (int): Dimension of states.
        action_dim (int): Dimension of actions.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(PolicyNet, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def get_actions(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        assert self._shape_check(
            states=states,
        ), f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."

        actions = self._get_actions_impl(states, exploring)

        assert self._shape_check(
            states=states,
            actions=actions,
        ), f"Actions shape check failed. Expecting: {(states.shape[0], self.action_dim)}, actual: {actions.shape}."

        return actions

    def get_actions_with_probs(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._shape_check(
            states=states,
        ), f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."

        actions, probs = self._get_actions_with_probs_impl(states, exploring)

        assert self._shape_check(
            states=states,
            actions=actions,
        ), f"Actions shape check failed. Expecting: {(states.shape[0], self.action_dim)}, actual: {actions.shape}."
        assert len(probs.shape) == 1 and probs.shape[0] == states.shape[0]

        return actions, probs

    def get_actions_with_logps(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._shape_check(
            states=states,
        ), f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."

        actions, logps = self._get_actions_with_logps_impl(states, exploring)

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
    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_actions_with_probs_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _get_actions_with_logps_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _get_states_actions_probs_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _shape_check(self, states: torch.Tensor, actions: torch.Tensor = None) -> bool:
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


class DiscretePolicyNet(PolicyNet, metaclass=ABCMeta):
    """Policy network for discrete action spaces.

    Args:
        state_dim (int): Dimension of states.
        action_num (int): Number of actions.
    """

    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscretePolicyNet, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    @property
    def action_num(self) -> int:
        return self._action_num

    def get_action_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Get the probabilities for all possible actions in the action space.

        Args:
            states (torch.Tensor): States.

        Returns:
            action_probs (torch.Tensor): Probability matrix with shape [batch_size, action_num].
        """
        assert self._shape_check(
            states=states,
        ), f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."
        action_probs = self._get_action_probs_impl(states)
        assert match_shape(action_probs, (states.shape[0], self.action_num)), (
            f"Action probabilities shape check failed. Expecting: {(states.shape[0], self.action_num)}, "
            f"actual: {action_probs.shape}."
        )
        return action_probs

    @abstractmethod
    def _get_action_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        """Implementation of `get_action_probs`. The core logic of a discrete policy net should be implemented here."""
        raise NotImplementedError

    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        actions, _ = self._get_actions_with_probs_impl(states, exploring)
        return actions

    def _get_actions_with_probs_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.get_action_probs(states)
        if exploring:
            distribution = Categorical(probs)
            actions = distribution.sample().unsqueeze(1)
            return actions, probs.gather(1, actions).squeeze(-1)
        else:
            probs, actions = probs.max(dim=1)
            return actions.unsqueeze(1), probs

    def _get_actions_with_logps_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        actions, probs = self._get_actions_with_probs_impl(states, exploring)
        return actions, torch.log(probs)

    def _get_states_actions_probs_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        probs = self.get_action_probs(states)
        return probs.gather(1, actions).squeeze(-1)

    def _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        probs = self._get_states_actions_probs_impl(states, actions)
        return torch.log(probs)


class ContinuousPolicyNet(PolicyNet, metaclass=ABCMeta):
    """Policy network for continuous action spaces.

    Args:
        state_dim (int): Dimension of states.
        action_dim (int): Dimension of actions.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(ContinuousPolicyNet, self).__init__(state_dim=state_dim, action_dim=action_dim)
