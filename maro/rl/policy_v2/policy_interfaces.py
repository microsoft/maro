from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

from maro.rl.utils import match_shape

"""
Mixins for policies.

Mixins have only methods (abstract or non abstract) which define a set of functions that a type of policies should have.
Abstract methods should be implemented by lower-level mixins or policy classes that inherit the mixin.

A policy class could inherit multiple mixins so that the combination of mixins determines the entire set of methods
of this policy.
"""


class DiscreteActionMixin:
    """Mixin for policies that generate discrete actions."""

    @property
    def action_num(self) -> int:
        return self._get_action_num()

    @abstractmethod
    def _get_action_num(self) -> int:
        pass


class MultiDiscreteActionMixin:
    """Mixin for multi-agent policies that generate discrete actions."""

    @property
    def action_nums(self) -> List[int]:
        return self._get_action_nums()

    @abstractmethod
    def _get_action_nums(self) -> List[int]:
        pass


class ContinuousActionMixin:
    """Mixin for policies that generate continuous actions."""

    @abstractmethod
    def action_range(self) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        return self._get_action_range()

    @abstractmethod
    def _get_action_range(self) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        pass


class ShapeCheckMixin:
    """
    Mixin that contains the `_shape_check` method, which is used for checking whether the states and actions
    have valid shapes. Usually, it should contains three parts:
    1. Check of states' shape.
    2. Check of actions' shape.
    3. Check whether states and actions have identical batch sizes.

    `actions` is optional. If it is None, it means we do not need to check action related issues.
    """

    @abstractmethod
    def _shape_check(self, states: np.ndarray, actions: Optional[np.ndarray]) -> bool:
        pass


class QNetworkMixin(ShapeCheckMixin):
    """
    Mixin for policies that have a Q-network in it, no matter how it is used. For example,
    both DQN policies and Actor-Critic policies that use a Q-network as the critic should inherit this mixin.
    """

    def q_values(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Returns Q-values based on given states and actions.
        The actual logics should be implemented in `_get_q_values`.

        Args:
            states (np.ndarray): states with shape [batch_size, state_dim]
            actions (np.ndarray): actions with shape [batch_size, action_dim]

        Returns:
            Q-values (np.ndarray) with shape [batch_size]
        """
        assert self._shape_check(states, actions)
        ret = self._get_q_values(states, actions)
        assert match_shape(ret, (states.shape[0],))
        return ret

    @abstractmethod
    def _get_q_values(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Implementation of `q_values`."""
        pass


class DiscreteQNetworkMixin(DiscreteActionMixin, QNetworkMixin):
    """Combination of DiscreteActionMixin and QNetworkMixin."""

    def q_values_for_all_actions(self, states: np.ndarray) -> np.ndarray:
        """
        Returns Q-values for all actions based on given states.
        The actual logics should be implemented in `_get_q_values_for_all_actions`.

        Args:
            states (np.ndarray): States with shape [batch_size, state_dim]

        Returns:
            Q-values (np.ndarray) with shape [batch_size, action_num]
        """
        assert self._shape_check(states, None)
        ret = self._get_q_values_for_all_actions(states)
        assert match_shape(ret, (states.shape[0], self.action_num))
        return ret

    @abstractmethod
    def _get_q_values_for_all_actions(self, states: np.ndarray) -> np.ndarray:
        """Implementation of `q_values_for_all_actions`."""
        pass


class VNetworkMixin(ShapeCheckMixin):
    """Mixin for policies that have a V-network in it. Similar to QNetworkMixin."""

    def v_values(self, states: np.ndarray) -> np.ndarray:
        """
        Returns Q-values based on given states.
        The actual logics should be implemented in `_get_v_values`.

        Args:
            states (np.ndarray): [batch_size, state_dim]

        Returns:
            V-values (np.ndarray): [batch_size]
        """
        assert self._shape_check(states, None)
        ret = self._get_v_values(states)
        assert match_shape(ret, (states.shape[0],))
        return ret

    @abstractmethod
    def _get_v_values(self, states: np.ndarray) -> np.ndarray:
        """Implementation of `v_values`."""
        pass
