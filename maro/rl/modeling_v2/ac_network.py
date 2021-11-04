from abc import ABCMeta
from typing import Optional, Tuple, Union

import torch
from torch.distributions import Categorical

from .base_model import ContinuousPolicyNetworkMixin, DiscreteProbPolicyNetworkMixin, PolicyNetwork
from .critic_model import CriticMixin, QCriticMixin, VCriticMixin


class ActorCriticNet(CriticMixin, PolicyNetwork, metaclass=ABCMeta):
    """
    Model framework for the actor-critic architecture.

    All concrete classes that inherit `ActorCriticNet` should implement the following abstract methods:
    - Declared in `AbsCoreModel`:
        - step(self, loss: torch.tensor) -> None:
        - get_gradients(self, loss: torch.tensor) -> torch.tensor:
        - apply_gradients(self, grad: dict) -> None:
        - get_state(self) -> object:
        - set_state(self, state: object) -> None:
    - Declared in `PolicyNetwork`:
        - _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
    """
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(ActorCriticNet, self).__init__(state_dim=state_dim, action_dim=action_dim)

    def _critic_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
        return self._policy_net_shape_check(states=states, actions=actions)


class DiscreteActorCriticNet(DiscreteProbPolicyNetworkMixin, ActorCriticNet, metaclass=ABCMeta):
    """
    Model framework for the actor-critic architecture for finite and discrete action spaces.

    All concrete classes that inherit `DiscreteActorCriticNet` should implement the following abstract methods:
    - Declared in `AbsCoreModel`:
        - step(self, loss: torch.tensor) -> None:
        - get_gradients(self, loss: torch.tensor) -> torch.tensor:
        - apply_gradients(self, grad: dict) -> None:
        - get_state(self) -> object:
        - set_state(self, state: object) -> None:
    - Declared in `DiscreteProbPolicyNetworkMixin`:
        - _get_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
    """

    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscreteActorCriticNet, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    def _get_action_num(self) -> int:
        return self._action_num

    def _get_actions_and_logps_exploring_impl(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs = Categorical(self.get_probs(states))
        actions = action_probs.sample()
        logps = action_probs.log_prob(actions)
        return actions, logps

    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        return self.get_actions_and_logps(states, exploring)[0].unsqueeze(1)


class DiscreteVActorCriticNet(VCriticMixin, DiscreteActorCriticNet, metaclass=ABCMeta):
    """
    Model framework for the actor-critic architecture for finite and discrete action spaces with V-value critic.

    All concrete classes that inherit `DiscreteVActorCriticNet` should implement the following abstract methods:
    - Declared in `AbsCoreModel`:
        - step(self, loss: torch.tensor) -> None:
        - get_gradients(self, loss: torch.tensor) -> torch.tensor:
        - apply_gradients(self, grad: dict) -> None:
        - get_state(self) -> object:
        - set_state(self, state: object) -> None:
    - Declared in `DiscreteProbPolicyNetworkMixin`:
        - _get_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
    - Declared in `VCriticMixin`:
        - _get_v_critic(self, states: torch.Tensor) -> torch.Tensor:
    """
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscreteVActorCriticNet, self).__init__(state_dim=state_dim, action_num=action_num)


class ContinuousActorCriticNet(ContinuousPolicyNetworkMixin, ActorCriticNet, metaclass=ABCMeta):
    """
    Model framework for the actor-critic architecture for continuous action spaces.

    All concrete classes that inherit `ContinuousActorCriticNet` should implement the following abstract methods:
    - Declared in `AbsCoreModel`:
        - step(self, loss: torch.tensor) -> None:
        - get_gradients(self, loss: torch.tensor) -> torch.tensor:
        - apply_gradients(self, grad: dict) -> None:
        - get_state(self) -> object:
        - set_state(self, state: object) -> None:
    - Declared in `PolicyNetwork`:
        - _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_lower_bound: Union[None, int, float],
        action_upper_bound: Union[None, int, float]
    ) -> None:
        super(ContinuousActorCriticNet, self).__init__(state_dim=state_dim, action_dim=action_dim)

        self._action_lower_bound = float('-inf') if action_lower_bound is None else float(action_lower_bound)
        self._action_upper_bound = float('inf') if action_upper_bound is None else float(action_upper_bound)
        assert self._action_lower_bound < self._action_upper_bound  # [lower, upper)

    def _get_action_range(self) -> Tuple[float, float]:
        return self._action_lower_bound, self._action_upper_bound


class ContinuousQActorCriticNet(QCriticMixin, ContinuousActorCriticNet, metaclass=ABCMeta):
    """
    Model framework for the actor-critic architecture for continuous action spaces with Q-value critic.

    All concrete classes that inherit `ContinuousQActorCriticNet` should implement the following abstract methods:
    - Declared in `AbsCoreModel`:
        - step(self, loss: torch.tensor) -> None:
        - get_gradients(self, loss: torch.tensor) -> torch.tensor:
        - apply_gradients(self, grad: dict) -> None:
        - get_state(self) -> object:
        - set_state(self, state: object) -> None:
    - Declared in `PolicyNetwork`:
        - _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
    - Declared in `QCriticMixin`:
        - _get_q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(ContinuousQActorCriticNet, self).__init__(state_dim=state_dim, action_dim=action_dim)

    def value(self, states: torch.Tensor) -> torch.Tensor:
        """
        [batch_size, state_dim] => [batch_size]
        """
        actions = self.get_actions(states, exploring=False)
        return self.q_critic(states, actions)
