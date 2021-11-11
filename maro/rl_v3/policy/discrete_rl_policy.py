from abc import ABCMeta
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from maro.rl_v3.model import DiscretePolicyNet, DiscreteQNet
from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.utils import match_shape


class DiscreteRLPolicy(RLPolicy, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        state_dim: int,
        action_num: int,
        device: str = None,
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
    """
    Valued-based policy.
    """
    def __init__(
        self,
        name: str,
        q_net: DiscreteQNet,
        device: str = None,
        trainable: bool = True
    ) -> None:
        assert isinstance(q_net, DiscreteQNet)

        super(ValueBasedPolicy, self).__init__(
            name=name, state_dim=q_net.state_dim, action_num=q_net.action_num, device=device, trainable=trainable
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
        assert match_shape(q_values, (states.shape[0],))  # [B]
        return q_values

    def explore(self) -> None:
        pass  # Overwrite the base method and turn off explore mode.

    def get_values_by_states_and_actions(self, states: np.ndarray, actions: np.ndarray) -> Optional[np.ndarray]:
        return self.q_values(states, actions)

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

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
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
    """
    Policy gradient policy that generates discrete actions.
    """
    def __init__(
        self,
        name: str,
        policy_net: DiscretePolicyNet,
        device: str = None,
        trainable: bool = True
    ) -> None:
        assert isinstance(policy_net, DiscretePolicyNet)

        super(DiscretePolicyGradient, self).__init__(
            name=name, state_dim=policy_net.state_dim, action_num=policy_net.action_num,
            device=device, trainable=trainable
        )

        self._policy_net = policy_net

    @property
    def policy_net(self) -> DiscretePolicyNet:
        return self._policy_net

    def get_values_by_states_and_actions(self, states: np.ndarray, actions: np.ndarray) -> Optional[np.ndarray]:
        return None  # PG policy does not have state values

    def _get_actions_with_logps_impl(
        self, states: torch.Tensor, exploring: bool, require_logps: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._policy_net.get_actions_with_logps(states, exploring, require_logps)

    def step(self, loss: torch.Tensor) -> None:
        self._policy_net.step(loss)

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        print(f'Huoran: set_policy_state, policy_name = {self.name}')
        self._policy_net.set_net_state(policy_state)

    def soft_update(self, other_policy: RLPolicy, tau: float) -> None:
        assert isinstance(other_policy, DiscretePolicyGradient)
        self._policy_net.soft_update(other_policy.policy_net, tau)

    def get_action_probs(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get the probabilities for all actions according to states.

        Args:
            states (torch.Tensor): States.

        Returns:
            Action probabilities with shape [batch_size, action_num]
        """
        assert self._shape_check(states=states), \
            f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."
        action_probs = self._policy_net.get_action_probs(states)
        assert match_shape(action_probs, (states.shape[0], self.action_num)), \
            f"Action probabilities shape check failed. Expecting: {(states.shape[0], self.action_num)}, " \
            f"actual: {action_probs.shape}."
        return action_probs
