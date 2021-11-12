from abc import ABCMeta, abstractmethod
from typing import List, Optional

import torch

from maro.rl.modeling_v2.base_model import AbsCoreModel
from maro.rl.utils import match_shape


class CriticMixin:
    """
    Mixin for all networks that used as critic models.
    """
    @abstractmethod
    def _critic_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
        """
        Checks whether the states and actions
        have valid shapes. Usually, it should contains three parts:
        1. Check of states' shape.
        2. Check of actions' shape.
        3. Check whether states and actions have identical batch sizes.

        Args:
            states (torch.Tensor): States with shape [batch_size, state_dim].
            actions (Optional[torch.Tensor]): Actions with shape [batch_size, action_dim] or None. If it is None, it
                means we don't need to check action related issues.

        Returns:
            Whether the states and actions have valid shapes
        """
        raise NotImplementedError


class VCriticMixin(CriticMixin):
    """
    Mixin for all networks that used as V-value based critic models.

    All concrete classes that inherit `VCriticMixin` should implement the following abstract methods:
    - Declared in `CriticMixin`:
        - _critic_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
    - Declared in `VCriticMixin`:
        - _get_v_critic(self, states: torch.Tensor) -> torch.Tensor:
    """
    def v_critic(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get V-values of the given states. The actual logics should be implemented in `_get_v_critic`.

        Args:
            states (torch.Tensor): States with shape [batch_size, state_dim].

        Returns:
            V-values for critic with shape [batch_size]
        """
        assert self._critic_net_shape_check(states=states, actions=None)
        ret = self._get_v_critic(states)
        assert match_shape(ret, (states.shape[0],))
        return ret

    @abstractmethod
    def _get_v_critic(self, states: torch.Tensor) -> torch.Tensor:
        """Implementation of v_critic."""
        pass


class QCriticMixin(CriticMixin):
    """
    Mixin for all networks that used as Q-value based critic models.

    All concrete classes that inherit `QCriticMixin` should implement the following abstract methods:
    - Declared in `CriticMixin`:
        - _critic_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
    - Declared in `QCriticMixin`:
        - _get_q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    def q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values according to the given states and actions.
        The actual logics should be implemented in `q_critic`.

        Args:
            states (torch.Tensor): States with shape [batch_size, state_dim].
            actions (torch.Tensor): Actions with shape [batch_size, action_dim].

        Returns:
            Q-values for critic with shape [batch_size]
        """
        #assert self._critic_net_shape_check(states=states, actions=actions)
        ret = self._get_q_critic(states, actions)
        assert match_shape(ret, (states.shape[0],))
        return ret

    @abstractmethod
    def _get_q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Implementation of q_critic."""
        pass


class CriticNetwork(AbsCoreModel, metaclass=ABCMeta):
    """
    Neural networks for critic models.

    All concrete classes that inherit `CriticNetwork` should implement the following abstract methods:
    - Declared in `AbsCoreModel`:
        - step(self, loss: torch.tensor) -> None:
        - get_gradients(self, loss: torch.tensor) -> torch.tensor:
        - apply_gradients(self, grad: dict) -> None:
        - get_state(self) -> object:
        - set_state(self, state: object) -> None:
    """
    def __init__(self, state_dim: int) -> None:
        super(CriticNetwork, self).__init__()
        self._state_dim = state_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def _is_valid_state_shape(self, states: torch.Tensor) -> bool:
        return states.shape[0] > 0 and match_shape(states, (None, self.state_dim))


class VCriticNetwork(VCriticMixin, CriticNetwork, metaclass=ABCMeta):
    """
    Neural networks for V-value based critic models.

    All concrete classes that inherit `VCriticNetwork` should implement the following abstract methods:
    - Declared in `AbsCoreModel`:
        - step(self, loss: torch.tensor) -> None:
        - get_gradients(self, loss: torch.tensor) -> torch.tensor:
        - apply_gradients(self, grad: dict) -> None:
        - get_state(self) -> object:
        - set_state(self, state: object) -> None:
    - Declared in `VCriticMixin`:
        - _get_v_critic(self, states: torch.Tensor) -> torch.Tensor:
    """
    def __init__(self, state_dim: int) -> None:
        super(VCriticNetwork, self).__init__(state_dim=state_dim)

    def _critic_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
        return self._is_valid_state_shape(states)


class QCriticNetwork(QCriticMixin, CriticNetwork, metaclass=ABCMeta):
    """
    Neural networks for Q-value based critic models.

    All concrete classes that inherit `QCriticNetwork` should implement the following abstract methods:
    - Declared in `AbsCoreModel`:
        - step(self, loss: torch.tensor) -> None:
        - get_gradients(self, loss: torch.tensor) -> torch.tensor:
        - apply_gradients(self, grad: dict) -> None:
        - get_state(self) -> object:
        - set_state(self, state: object) -> None:
    - Declared in `QCriticMixin`:
        - _get_q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(QCriticNetwork, self).__init__(state_dim=state_dim)
        self._action_dim = action_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _critic_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
        return all([
            self._is_valid_state_shape(states),
            self._is_valid_action_shape(actions),
            states.shape[0] == actions.shape[0],
        ])

    def _is_valid_action_shape(self, actions: torch.Tensor) -> bool:
        return actions.shape[0] > 0 and match_shape(actions, (None, self.action_dim))


class DiscreteQCriticNetwork(QCriticNetwork):
    """
    Neural networks for Q-value based critic models that take discrete actions as inputs.

    All concrete classes that inherit `DiscreteQCriticNetwork` should implement the following abstract methods:
    - Declared in `AbsCoreModel`:
        - step(self, loss: torch.tensor) -> None:
        - get_gradients(self, loss: torch.tensor) -> torch.tensor:
        - apply_gradients(self, grad: dict) -> None:
        - get_state(self) -> object:
        - set_state(self, state: object) -> None:
    - Declared in `QCriticMixin`:
        - _get_q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    - Declared in `DiscreteQCriticNetwork`:
        - _get_q_critic_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
    """
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscreteQCriticNetwork, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    def _get_q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q_values = self.q_critic_for_all_actions(states, actions)  # [batch_size, action_num]
        return q_values.reshape(-1)

    @property
    def action_num(self) -> int:
        return self._action_num

    def q_critic_for_all_actions(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Generates the matrix that contains the Q-values for all potential actions.
        The actual logics should be implemented in `_get_q_critic_for_all_actions`.

        Args:
            states (torch.Tensor): States with shape [batch_size, state_dim].

        Returns:
            q critics for all actions with shape [batch_size, action_num]
        """
        #assert self._is_valid_state_shape(states)
        ret = self._get_q_critic_for_all_actions(states, actions)
        assert match_shape(ret, (states.shape[0], self.action_num))
        return ret

    @abstractmethod
    def _get_q_critic_for_all_actions(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Implementation of `q_critic_for_all_actions`"""
        pass


class MultiQCriticNetwork(QCriticMixin, CriticNetwork, metaclass=ABCMeta):
    """
    Neural networks for Q-value based critic models that takes multiple actions as inputs.
    This is used for multi-agent RL scenarios.

    All concrete classes that inherit `MultiQCriticNetwork` should implement the following abstract methods:
    - Declared in `AbsCoreModel`:
        - step(self, loss: torch.tensor) -> None:
        - get_gradients(self, loss: torch.tensor) -> torch.tensor:
        - apply_gradients(self, grad: dict) -> None:
        - get_state(self) -> object:
        - set_state(self, state: object) -> None:
    - Declared in `QCriticMixin`:
        - _get_q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    def __init__(self, state_dim: int, action_dim: int, agent_num: int) -> None:
        super(MultiQCriticNetwork, self).__init__(state_dim=state_dim)
        self._action_dim = action_dim
        self._agent_num = agent_num

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def agent_num(self) -> int:
        return self._agent_num

    def _critic_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
        return all([
            self._is_valid_state_shape(states),
            actions is None or self._is_valid_action_shape(actions),
            actions is None or states.shape[0] == actions.shape[0]
        ])

    def _is_valid_action_shape(self, actions: torch.Tensor) -> bool:
        return match_shape(actions, (None, self.agent_num, self.action_dim))


class MultiDiscreteQCriticNetwork(MultiQCriticNetwork, metaclass=ABCMeta):
    """
    Neural networks for Q-value based critic models that take multiple discrete actions as inputs.

    All concrete classes that inherit `MultiDiscreteQCriticNetwork` should implement the following abstract methods:
    - Declared in `AbsCoreModel`:
        - step(self, loss: torch.tensor) -> None:
        - get_gradients(self, loss: torch.tensor) -> torch.tensor:
        - apply_gradients(self, grad: dict) -> None:
        - get_state(self) -> object:
        - set_state(self, state: object) -> None:
    - Declared in `QCriticMixin`:
        - _get_q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    def __init__(self, state_dim: int, action_nums: List[int]) -> None:
        super(MultiDiscreteQCriticNetwork, self).__init__(state_dim=state_dim, action_dim=1, agent_num=len(action_nums))
        self._action_nums = action_nums
        self._agent_num = len(action_nums)

    @property
    def action_nums(self) -> List[int]:
        return self._action_nums

    @property
    def agent_num(self) -> int:
        return self._agent_num
