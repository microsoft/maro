# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from maro.rl.exploration import ExploreStrategy
from maro.rl.model import DiscretePolicyNet, DiscreteQNet
from maro.rl.utils import match_shape, ndarray_to_tensor

from .abs_policy import RLPolicy


class DiscreteRLPolicy(RLPolicy, metaclass=ABCMeta):
    """RL policy for discrete action spaces.

    Args:
        name (str): Name of the policy.
        state_dim (int): Dimension of states.
        action_num (int): Number of actions.
        trainable (bool, default=True): Whether this policy is trainable.
        warmup (int, default=0): Number of steps for uniform-random action selection, before running real policy.
            Helps exploration.
    """

    def __init__(
        self,
        name: str,
        state_dim: int,
        action_num: int,
        trainable: bool = True,
        warmup: int = 0,
    ) -> None:
        assert action_num >= 1

        super(DiscreteRLPolicy, self).__init__(
            name=name,
            state_dim=state_dim,
            action_dim=1,
            trainable=trainable,
            is_discrete_action=True,
            warmup=warmup,
        )

        self._action_num = action_num

    @property
    def action_num(self) -> int:
        return self._action_num

    def _post_check(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> bool:
        return all([0 <= action < self.action_num for action in actions.cpu().numpy().flatten()])

    def _get_random_actions_impl(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        return ndarray_to_tensor(
            np.random.randint(self.action_num, size=(states.shape[0], 1)),
            device=self._device,
        )


class ValueBasedPolicy(DiscreteRLPolicy):
    """Valued-based policy.

    Args:
        name (str): Name of the policy.
        q_net (DiscreteQNet): Q-net used in this value-based policy.
        trainable (bool, default=True): Whether this policy is trainable.
        explore_strategy (Optional[ExploreStrategy], default=None): Explore strategy.
        warmup (int, default=50000): Number of steps for uniform-random action selection, before running real policy.
            Helps exploration.
    """

    def __init__(
        self,
        name: str,
        q_net: DiscreteQNet,
        trainable: bool = True,
        explore_strategy: Optional[ExploreStrategy] = None,
        warmup: int = 50000,
    ) -> None:
        assert isinstance(q_net, DiscreteQNet)

        super(ValueBasedPolicy, self).__init__(
            name=name,
            state_dim=q_net.state_dim,
            action_num=q_net.action_num,
            trainable=trainable,
            warmup=warmup,
        )
        self._q_net = q_net
        self._explore_strategy = explore_strategy
        self._softmax = torch.nn.Softmax(dim=1)

    @property
    def q_net(self) -> DiscreteQNet:
        return self._q_net

    def q_values_for_all_actions(self, states: np.ndarray, **kwargs) -> np.ndarray:
        """Generate a matrix containing the Q-values for all actions for the given states.

        Args:
            states (np.ndarray): States.

        Returns:
            q_values (np.ndarray): Q-matrix.
        """
        return (
            self.q_values_for_all_actions_tensor(
                ndarray_to_tensor(states, device=self._device),
                **kwargs,
            )
            .cpu()
            .numpy()
        )

    def q_values_for_all_actions_tensor(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate a matrix containing the Q-values for all actions for the given states.

        Args:
            states (torch.Tensor): States.

        Returns:
            q_values (torch.Tensor): Q-matrix.
        """
        assert self._shape_check(states=states, **kwargs)
        q_values = self._q_net.q_values_for_all_actions(states, **kwargs)
        assert match_shape(q_values, (states.shape[0], self.action_num))  # [B, action_num]
        return q_values

    def q_values(self, states: np.ndarray, actions: np.ndarray, **kwargs) -> np.ndarray:
        """Generate the Q values for given state-action pairs.

        Args:
            states (np.ndarray): States.
            actions (np.ndarray): Actions. Should has same length with states.

        Returns:
            q_values (np.ndarray): Q-values.
        """
        return (
            self.q_values_tensor(
                ndarray_to_tensor(states, device=self._device),
                ndarray_to_tensor(actions, device=self._device),
                **kwargs,
            )
            .cpu()
            .numpy()
        )

    def q_values_tensor(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate the Q values for given state-action pairs.

        Args:
            states (torch.Tensor): States.
            actions (torch.Tensor): Actions. Should has same length with states.

        Returns:
            q_values (torch.Tensor): Q-values.
        """
        assert self._shape_check(states=states, actions=actions, **kwargs)  # actions: [B, 1]
        q_values = self._q_net.q_values(states, actions, **kwargs)
        assert match_shape(q_values, (states.shape[0],))  # [B]
        return q_values

    def _get_actions_impl(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._get_actions_with_probs_impl(states, **kwargs)[0]

    def _get_actions_with_probs_impl(self, states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        q_matrix = self.q_values_for_all_actions_tensor(states, **kwargs)  # [B, action_num]
        q_matrix_softmax = self._softmax(q_matrix)
        _, actions = q_matrix.max(dim=1)  # [B], [B]

        if self._is_exploring and self._explore_strategy is not None:
            actions = self._explore_strategy.get_action(state=states.cpu().numpy(), action=actions.cpu().numpy())
            actions = ndarray_to_tensor(actions, device=self._device)

        actions = actions.unsqueeze(1).long()
        return actions, q_matrix_softmax.gather(1, actions).squeeze(-1)  # [B, 1]

    def _get_actions_with_logps_impl(self, states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        actions, probs = self._get_actions_with_probs_impl(states, **kwargs)
        return actions, torch.log(probs)

    def _get_states_actions_probs_impl(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        q_matrix = self.q_values_for_all_actions_tensor(states, **kwargs)
        q_matrix_softmax = self._softmax(q_matrix)
        return q_matrix_softmax.gather(1, actions).squeeze(-1)  # [B]

    def _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        probs = self._get_states_actions_probs_impl(states, actions, **kwargs)
        return torch.log(probs)

    def train_step(self, loss: torch.Tensor) -> None:
        return self._q_net.step(loss)

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._q_net.get_gradients(loss)

    def apply_gradients(self, grad: dict) -> None:
        self._q_net.apply_gradients(grad)

    def freeze(self) -> None:
        self._q_net.freeze()

    def unfreeze(self) -> None:
        self._q_net.unfreeze()

    def eval(self) -> None:
        self._q_net.eval()

    def train(self) -> None:
        self._q_net.train()

    def get_state(self) -> dict:
        return {
            "net": self._q_net.get_state(),
            "policy": {
                "warmup": self._warmup,
                "call_count": self._call_count,
            },
        }

    def set_state(self, policy_state: dict) -> None:
        self._q_net.set_state(policy_state["net"])
        self._warmup = policy_state["policy"]["warmup"]
        self._call_count = policy_state["policy"]["call_count"]

    def soft_update(self, other_policy: RLPolicy, tau: float) -> None:
        assert isinstance(other_policy, ValueBasedPolicy)
        self._q_net.soft_update(other_policy.q_net, tau)

    def _to_device_impl(self, device: torch.device) -> None:
        self._q_net.to_device(device)


class DiscretePolicyGradient(DiscreteRLPolicy):
    """Policy gradient for discrete action spaces.

    Args:
        name (str): Name of the policy.
        policy_net (DiscretePolicyNet): The core net of this policy.
        trainable (bool, default=True): Whether this policy is trainable.
        warmup (int, default=50000): Number of steps for uniform-random action selection, before running real policy.
            Helps exploration.
    """

    def __init__(
        self,
        name: str,
        policy_net: DiscretePolicyNet,
        trainable: bool = True,
        warmup: int = 0,
    ) -> None:
        assert isinstance(policy_net, DiscretePolicyNet)

        super(DiscretePolicyGradient, self).__init__(
            name=name,
            state_dim=policy_net.state_dim,
            action_num=policy_net.action_num,
            trainable=trainable,
            warmup=warmup,
        )

        self._policy_net = policy_net

    @property
    def policy_net(self) -> DiscretePolicyNet:
        return self._policy_net

    def _get_actions_impl(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._policy_net.get_actions(states, self._is_exploring, **kwargs)

    def _get_actions_with_probs_impl(self, states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._policy_net.get_actions_with_probs(states, self._is_exploring, **kwargs)

    def _get_actions_with_logps_impl(self, states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._policy_net.get_actions_with_logps(states, self._is_exploring, **kwargs)

    def _get_states_actions_probs_impl(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._policy_net.get_states_actions_probs(states, actions, **kwargs)

    def _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._policy_net.get_states_actions_logps(states, actions, **kwargs)

    def train_step(self, loss: torch.Tensor) -> None:
        self._policy_net.step(loss)

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._policy_net.get_gradients(loss)

    def apply_gradients(self, grad: dict) -> None:
        self._policy_net.apply_gradients(grad)

    def freeze(self) -> None:
        self._policy_net.freeze()

    def unfreeze(self) -> None:
        self._policy_net.unfreeze()

    def eval(self) -> None:
        self._policy_net.eval()

    def train(self) -> None:
        self._policy_net.train()

    def get_state(self) -> dict:
        return {
            "net": self._policy_net.get_state(),
            "policy": {
                "warmup": self._warmup,
                "call_count": self._call_count,
            },
        }

    def set_state(self, policy_state: dict) -> None:
        self._policy_net.set_state(policy_state["net"])
        self._warmup = policy_state["policy"]["warmup"]
        self._call_count = policy_state["policy"]["call_count"]

    def soft_update(self, other_policy: RLPolicy, tau: float) -> None:
        assert isinstance(other_policy, DiscretePolicyGradient)
        self._policy_net.soft_update(other_policy.policy_net, tau)

    def get_action_probs(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get the probabilities for all actions according to states.

        Args:
            states (torch.Tensor): States.

        Returns:
            action_probs (torch.Tensor): Action probabilities with shape [batch_size, action_num].
        """
        assert self._shape_check(
            states=states,
            **kwargs,
        ), f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."
        action_probs = self._policy_net.get_action_probs(states, **kwargs)
        assert match_shape(action_probs, (states.shape[0], self.action_num)), (
            f"Action probabilities shape check failed. Expecting: {(states.shape[0], self.action_num)}, "
            f"actual: {action_probs.shape}."
        )
        return action_probs

    def get_action_logps(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get the log-probabilities for all actions according to states.

        Args:
            states (torch.Tensor): States.

        Returns:
            action_logps (torch.Tensor): Action probabilities with shape [batch_size, action_num].
        """
        return torch.log(self.get_action_probs(states, **kwargs))

    def _get_state_action_probs_impl(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        action_probs = self.get_action_probs(states, **kwargs)
        return action_probs.gather(1, actions).squeeze(-1)  # [B]

    def _get_state_action_logps_impl(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        action_logps = self.get_action_logps(states, **kwargs)
        return action_logps.gather(1, actions).squeeze(-1)  # [B]

    def _to_device_impl(self, device: torch.device) -> None:
        self._policy_net.to_device(device)
