from abc import abstractmethod

import torch

from .base_model import DiscretePolicyNetworkMixin, PolicyNetwork
from ..utils import match_shape


class QNetwork(PolicyNetwork):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(QNetwork, self).__init__(state_dim, action_dim)

    def q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return the Q-values according to states and actions.

        Args:
            states: shape = [batch_size, state_dim].
            actions: shape = [batch_size, action_dim].

        Returns:
            Q-values of shape [batch_size].
        """
        assert self._policy_net_shape_check(states, actions)
        ret = self._get_q_values(states, actions)
        assert match_shape(ret, (states.shape[0],))
        return ret

    @abstractmethod
    def _get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pass


class DiscreteQNetwork(DiscretePolicyNetworkMixin, QNetwork):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscreteQNetwork, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    def _get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q_matrix = self.q_values_for_all_actions(states)  # [batch_size, action_num]
        return q_matrix.gather(dim=1, index=actions).reshape(-1)

    def q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Return the matrix that contains Q-values for all possible actions.

        Args:
            states: shape = [batch_size, state_dim]

        Returns:
            Q-value matrix of shape [batch_size, action_num]
        """
        assert self._policy_net_shape_check(states, None)
        ret = self._get_q_values_for_all_actions(states)
        assert match_shape(ret, (states.shape[0], self.action_num))
        return ret

    @abstractmethod
    def _get_q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        pass

    def _get_action_num(self) -> int:
        return self._action_num

    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        if exploring:
            raise NotImplementedError
        else:
            q_matrix = self.q_values_for_all_actions(states)
            _, action = q_matrix.max(dim=1)
            return action.unsqueeze(1)
