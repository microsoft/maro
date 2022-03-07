# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod

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
        """Get actions according to the given states.

        Args:
            states (torch.Tensor): States.
            exploring (bool): If it is True, return the results under exploring mode. Otherwise, return the results
                under exploiting mode.

        Returns:
            Actions (torch.Tensor)
        """
        assert self._shape_check(states=states), \
            f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."
        actions = self._get_actions_impl(states, exploring)
        assert self._shape_check(states=states, actions=actions), \
            f"Actions shape check failed. Expecting: {(states.shape[0], self.action_dim)}, actual: {actions.shape}."
        return actions

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

    @abstractmethod
    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        """Implementation of `get_actions`.
        """
        raise NotImplementedError


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
        assert self._shape_check(states=states), \
            f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."
        action_probs = self._get_action_probs_impl(states)
        assert match_shape(action_probs, (states.shape[0], self.action_num)), \
            f"Action probabilities shape check failed. Expecting: {(states.shape[0], self.action_num)}, " \
            f"actual: {action_probs.shape}."
        return action_probs

    def get_action_logps(self, states: torch.Tensor) -> torch.Tensor:
        """Get the log-probabilities for all actions.

        Args:
            states (torch.Tensor): States.

        Returns:
            logps (torch.Tensor): Lop-probability matrix with shape [batch_size, action_num].
        """
        return torch.log(self.get_action_probs(states))

    @abstractmethod
    def _get_action_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        """Implementation of `get_action_probs`. The core logic of a discrete policy net should be implemented here.
        """
        raise NotImplementedError

    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        if exploring:
            actions = self._get_actions_exploring_impl(states)
            return actions
        else:
            action_logps = self.get_action_logps(states)
            logps, actions = action_logps.max(dim=1)
            return actions.unsqueeze(1)

    def _get_actions_exploring_impl(self, states: torch.Tensor) -> torch.Tensor:
        """Get actions according to the states under exploring mode.

        Args:
            states (torch.Tensor): States.

        Returns:
            actions (torch.Tensor): Actions.
        """
        action_probs = Categorical(self.get_action_probs(states))
        actions = action_probs.sample()
        return actions.unsqueeze(1)


class ContinuousPolicyNet(PolicyNet, metaclass=ABCMeta):
    """Policy network for continuous action spaces.

    Args:
        state_dim (int): Dimension of states.
        action_dim (int): Dimension of actions.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(ContinuousPolicyNet, self).__init__(state_dim=state_dim, action_dim=action_dim)
