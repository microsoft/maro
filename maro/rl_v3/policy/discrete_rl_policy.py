from abc import ABCMeta
from typing import Optional, Tuple

import numpy as np
import torch

from maro.rl_v3.model import DiscretePolicyNet
from maro.rl_v3.model.q_net import DiscreteQNet
from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.utils import match_shape


class DiscreteRLPolicy(RLPolicy, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        state_dim: int,
        action_num: int,
        device: str,
        trainable: bool = True
    ) -> None:
        assert action_num >= 1

        super(DiscreteRLPolicy, self).__init__(
            name=name, state_dim=state_dim, action_dim=1, device=device, trainable=trainable
        )

        self._action_num = action_num

    @property
    def action_num(self) -> int:
        return self._action_num

    def _post_check(self, states: torch.Tensor, actions: torch.Tensor) -> bool:
        return all([0 <= action < self.action_num for action in actions.cpu().numpy().flatten()])


class ValueBasedPolicy(DiscreteRLPolicy):
    def __init__(
        self,
        name: str,
        state_dim: int,
        action_num: int,
        q_net: DiscreteQNet,
        device: str,
        trainable: bool = True
    ) -> None:
        assert isinstance(q_net, DiscreteQNet)
        assert q_net.state_dim == self.state_dim
        assert q_net.action_dim == self.action_dim
        assert q_net.action_num == self.action_num

        super(ValueBasedPolicy, self).__init__(
            name=name, state_dim=state_dim, action_num=action_num, device=device, trainable=trainable
        )
        self._q_net = q_net

    @property
    def q_net(self) -> DiscreteQNet:
        return self._q_net

    def q_values_for_all_actions(self, states: np.ndarray) -> np.ndarray:
        return self.q_values_for_all_actions_tensor(self.ndarray_to_tensor(states)).cpu().numpy()

    def q_values_for_all_actions_tensor(self, states: torch.Tensor) -> torch.Tensor:
        assert self._shape_check(states=states)
        q_values = self._q_net.q_values_for_all_actions(states)
        assert match_shape(q_values, (states.shape[0], self.action_num))  # [B, action_num]
        return q_values

    def q_values(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.q_values_tensor(
            self.ndarray_to_tensor(states),
            self.ndarray_to_tensor(actions)
        ).cpu().numpy()

    def q_values_tensor(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        assert self._shape_check(states=states, actions=actions)  # actions: [B, 1]
        q_values = self._q_net.q_values(states, actions)
        assert match_shape(q_values, (states.shape[0], 1))  # [B, 1]
        return q_values

    def explore(self) -> None:
        raise NotImplementedError

    def _get_actions_with_logps_impl(
        self, states: torch.Tensor, exploring: bool, require_logps: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if exploring:
            raise NotImplementedError
        else:
            q_matrix = self.q_values_for_all_actions_tensor(states)  # [B, action_num]
            logps, action = q_matrix.max(dim=1)  # [B], [B]
            return action.unsqueeze(1), logps if require_logps else None  # [B, 1], [B]

    def step(self, loss: torch.Tensor) -> None:
        self._q_net.step(loss)

    def get_gradients(self, loss: torch.Tensor) -> torch.Tensor:
        return self._q_net.get_gradients(loss)

    def freeze(self) -> None:
        self._q_net.freeze()

    def unfreeze(self) -> None:
        self._q_net.unfreeze()

    def eval(self) -> None:
        self._q_net.eval()

    def train(self) -> None:
        self._q_net.train()

    def get_policy_state(self) -> object:
        return self._q_net.get_net_state()

    def set_policy_state(self, policy_state: object) -> None:
        self._q_net.set_net_state(policy_state)

    def soft_update(self, other_policy: RLPolicy, tau: float) -> None:
        assert isinstance(other_policy, ValueBasedPolicy)
        self._q_net.soft_update(other_policy.q_net, tau)


class DiscretePolicyGradient(DiscreteRLPolicy):
    def __init__(
        self,
        name: str,
        state_dim: int,
        action_num: int,
        policy_net: DiscretePolicyNet,
        device: str,
        trainable: bool = True
    ) -> None:
        assert isinstance(policy_net, DiscretePolicyNet)
        assert policy_net.state_dim == self.state_dim
        assert policy_net.action_dim == self.action_dim
        assert policy_net.action_num == self.action_num

        super(DiscretePolicyGradient, self).__init__(
            name=name, state_dim=state_dim, action_num=action_num, device=device, trainable=trainable
        )

        self._policy_net = policy_net

    @property
    def policy_net(self) -> DiscretePolicyNet:
        return self._policy_net

    def _get_actions_with_logps_impl(
        self, states: torch.Tensor, exploring: bool, require_logps: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._policy_net.get_actions_with_logps(states, exploring, require_logps)

    def step(self, loss: torch.Tensor) -> None:
        self._policy_net.step(loss)

    def get_gradients(self, loss: torch.Tensor) -> torch.Tensor:
        return self._policy_net.get_gradients(loss)

    def freeze(self) -> None:
        self._policy_net.freeze()

    def unfreeze(self) -> None:
        self._policy_net.unfreeze()

    def eval(self) -> None:
        self._policy_net.eval()

    def train(self) -> None:
        self._policy_net.train()

    def get_policy_state(self) -> object:
        return self._policy_net.get_net_state()

    def set_policy_state(self, policy_state: object) -> None:
        self._policy_net.set_net_state(policy_state)

    def soft_update(self, other_policy: RLPolicy, tau: float) -> None:
        assert isinstance(other_policy, DiscretePolicyGradient)
        self._policy_net.soft_update(other_policy.policy_net, tau)
