from abc import ABCMeta, abstractmethod
from typing import List, Optional

import torch

from maro.rl.modeling_v2.base_model import AbsCoreModel
from maro.rl.utils import match_shape


class CriticMixin:
    @abstractmethod
    def _critic_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
        pass


class VCriticMixin(CriticMixin):
    def v_critic(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: [batch_size, state_dim]

        Returns:
            v values for critic: [batch_size]
        """
        assert self._critic_net_shape_check(states, None)
        ret = self._get_v_critic(states)
        assert match_shape(ret, (states.shape[0],))
        return ret

    @abstractmethod
    def _get_v_critic(self, states: torch.Tensor) -> torch.Tensor:
        pass


class QCriticMixin(CriticMixin):
    @abstractmethod
    def q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: [batch_size, state_dim]
            actions:
                [batch_size, action_dim] for single agent
                [batch_size, sub_agent_num, action_dim] for multi agents

        Returns:
            q values for critic: [batch_size]
        """
        assert self._critic_net_shape_check(states, actions)
        ret = self._get_q_critic(states, actions)
        assert match_shape(ret, (states.shape[0],))
        return ret

    @abstractmethod
    def _get_q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pass


class CriticNetwork(AbsCoreModel, metaclass=ABCMeta):
    def __init__(self, state_dim: int) -> None:
        super(CriticNetwork, self).__init__()
        self._state_dim = state_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def _is_valid_state_shape(self, states: torch.Tensor) -> bool:
        return match_shape(states, (None, self.state_dim))


class VCriticNetwork(VCriticMixin, CriticNetwork, metaclass=ABCMeta):
    def __init__(self, state_dim: int) -> None:
        super(VCriticNetwork, self).__init__(state_dim=state_dim)

    def _critic_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
        return self._is_valid_state_shape(states)


class QCriticNetwork(QCriticMixin, CriticNetwork, metaclass=ABCMeta):
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
            states.shape[0] == actions.shape[0]
        ])

    def _is_valid_action_shape(self, actions: torch.Tensor) -> bool:
        return match_shape(actions, (None, self.action_dim))


class DiscreteQCriticNetwork(QCriticNetwork):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscreteQCriticNetwork, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    def _get_q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q_matrix = self.q_critic_for_all_actions(states)  # [batch_size, action_num]
        actions = actions.unsqueeze(dim=1)
        return q_matrix.gather(dim=1, index=actions).reshape(-1)

    @property
    def action_num(self) -> int:
        return self._action_num

    def q_critic_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: [batch_size, state_dim]

        Returns:
            q values for all actions: [batch_size, action_num]
        """
        assert self._is_valid_state_shape(states)
        ret = self._get_q_critic_for_all_actions(states)
        assert match_shape(ret, (states.shape[0], self.action_num))
        return ret

    @abstractmethod
    def _get_q_critic_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        pass


class MultiQCriticNetwork(QCriticMixin, CriticNetwork, metaclass=ABCMeta):
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
