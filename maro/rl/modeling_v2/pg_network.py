from abc import ABCMeta
from typing import Tuple

import torch
from torch.distributions import Categorical

from .base_model import DiscreteProbPolicyNetworkMixin, PolicyNetwork


class DiscretePolicyGradientNetwork(DiscreteProbPolicyNetworkMixin, PolicyNetwork, metaclass=ABCMeta):
    """Model framework for the policy gradient networks."""

    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscretePolicyGradientNetwork, self).__init__(state_dim, 1)
        self._action_num = action_num

    def _get_actions_and_logps_exploring_impl(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs = Categorical(self.get_probs(states))
        actions = action_probs.sample()
        logps = action_probs.log_prob(actions)
        return actions, logps

    def _get_action_num(self) -> int:
        return self._action_num

    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        return self.get_actions_and_logps(states, exploring)[0].unsqueeze(1)
