from abc import ABCMeta
from typing import Optional, Tuple

import torch
from torch.distributions import Categorical

from .base_model import DiscreteProbPolicyNetworkMixin, PolicyNetwork
from .critic_model import VCriticMixin


class DiscreteActorCriticNet(DiscreteProbPolicyNetworkMixin, PolicyNetwork, metaclass=ABCMeta):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscreteActorCriticNet, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    def _get_action_num(self) -> int:
        return self._action_num

    def _get_actions_and_logps_exploration_impl(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs = Categorical(self.get_probs(states))
        actions = action_probs.sample()
        logps = action_probs.log_prob(actions)
        return actions, logps

    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        return self.get_actions_and_logps(states, exploring)[0].unsqueeze(1)


class DiscreteVActorCriticNet(VCriticMixin, DiscreteActorCriticNet, metaclass=ABCMeta):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscreteVActorCriticNet, self).__init__(state_dim=state_dim, action_num=action_num)

    def _critic_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
        return self._policy_net_shape_check(states, actions)
