# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from maro.rl.exploration import epsilon_greedy
from maro.rl.model import DiscretePolicyNet, DiscreteQNet
from maro.rl.utils import match_shape, ndarray_to_tensor
from maro.utils import clone

from .abs_policy import RLPolicy


class DiscreteRLPolicy(RLPolicy, metaclass=ABCMeta):
    """RL policy for discrete action spaces.

    Args:
        name (str): Name of the policy.
        state_dim (int): Dimension of states.
        action_num (int): Number of actions.
        trainable (bool, default=True): Whether this policy is trainable.
    """

    def __init__(
        self,
        name: str,
        state_dim: int,
        action_num: int,
        trainable: bool = True,
    ) -> None:
        assert action_num >= 1

        super(DiscreteRLPolicy, self).__init__(
            name=name,
            state_dim=state_dim,
            action_dim=1,
            trainable=trainable,
            is_discrete_action=True,
        )

        self._action_num = action_num

    @property
    def action_num(self) -> int:
        return self._action_num

    def _post_check(self, states: torch.Tensor, actions: torch.Tensor) -> bool:
        return all([0 <= action < self.action_num for action in actions.cpu().numpy().flatten()])


class ValueBasedPolicy(DiscreteRLPolicy):
    """Valued-based policy.

    Args:
        name (str): Name of the policy.
        q_net (DiscreteQNet): Q-net used in this value-based policy.
        trainable (bool, default=True): Whether this policy is trainable.
        exploration_strategy (Tuple[Callable, dict], default=(epsilon_greedy, {"epsilon": 0.1})): Exploration strategy.
        exploration_scheduling_options (List[tuple], default=None): List of exploration scheduler options.
        warmup (int, default=50000): Minimum number of experiences to warm up this policy.
    """

    def __init__(
        self,
        name: str,
        q_net: DiscreteQNet,
        trainable: bool = True,
        exploration_strategy: Tuple[Callable, dict] = (epsilon_greedy, {"epsilon": 0.1}),
        exploration_scheduling_options: List[tuple] = None,
        warmup: int = 50000,
    ) -> None:
        assert isinstance(q_net, DiscreteQNet)

        super(ValueBasedPolicy, self).__init__(
            name=name,
            state_dim=q_net.state_dim,
            action_num=q_net.action_num,
            trainable=trainable,
        )
        self._q_net = q_net

        self._exploration_func = exploration_strategy[0]
        self._exploration_params = clone(exploration_strategy[1])  # deep copy is needed to avoid unwanted sharing
        self._exploration_schedulers = (
            [opt[1](self._exploration_params, opt[0], **opt[2]) for opt in exploration_scheduling_options]
            if exploration_scheduling_options is not None
            else []
        )

        self._call_cnt = 0
        self._warmup = warmup

        self._softmax = torch.nn.Softmax(dim=1)

    @property
    def q_net(self) -> DiscreteQNet:
        return self._q_net

    def q_values_for_all_actions(self, states: np.ndarray) -> np.ndarray:
        """Generate a matrix containing the Q-values for all actions for the given states.

        Args:
            states (np.ndarray): States.

        Returns:
            q_values (np.ndarray): Q-matrix.
        """
        return self.q_values_for_all_actions_tensor(ndarray_to_tensor(states, device=self._device)).cpu().numpy()

    def q_values_for_all_actions_tensor(self, states: torch.Tensor) -> torch.Tensor:
        """Generate a matrix containing the Q-values for all actions for the given states.

        Args:
            states (torch.Tensor): States.

        Returns:
            q_values (torch.Tensor): Q-matrix.
        """
        assert self._shape_check(states=states)
        q_values = self._q_net.q_values_for_all_actions(states)
        assert match_shape(q_values, (states.shape[0], self.action_num))  # [B, action_num]
        return q_values

    def q_values(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
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
            )
            .cpu()
            .numpy()
        )

    def q_values_tensor(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Generate the Q values for given state-action pairs.

        Args:
            states (torch.Tensor): States.
            actions (torch.Tensor): Actions. Should has same length with states.

        Returns:
            q_values (torch.Tensor): Q-values.
        """
        assert self._shape_check(states=states, actions=actions)  # actions: [B, 1]
        q_values = self._q_net.q_values(states, actions)
        assert match_shape(q_values, (states.shape[0],))  # [B]
        return q_values

    def explore(self) -> None:
        pass  # Overwrite the base method and turn off explore mode.

    def _get_actions_impl(self, states: torch.Tensor) -> torch.Tensor:
        actions, _ = self._get_actions_with_probs_impl(states)
        return actions

    def _get_actions_with_probs_impl(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._call_cnt += 1
        if self._call_cnt <= self._warmup:
            actions = ndarray_to_tensor(
                np.random.randint(self.action_num, size=(states.shape[0], 1)),
                device=self._device,
            )
            probs = torch.ones(states.shape[0]).float() * (1.0 / self.action_num)
            return actions, probs

        q_matrix = self.q_values_for_all_actions_tensor(states)  # [B, action_num]
        q_matrix_softmax = self._softmax(q_matrix)
        _, actions = q_matrix.max(dim=1)  # [B], [B]

        if self._is_exploring:
            actions = self._exploration_func(states, actions.cpu().numpy(), self.action_num, **self._exploration_params)
            actions = ndarray_to_tensor(actions, device=self._device)

        actions = actions.unsqueeze(1)
        return actions, q_matrix_softmax.gather(1, actions).squeeze(-1)  # [B, 1]

    def _get_actions_with_logps_impl(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions, probs = self._get_actions_with_probs_impl(states)
        return actions, torch.log(probs)

    def _get_states_actions_probs_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q_matrix = self.q_values_for_all_actions_tensor(states)
        q_matrix_softmax = self._softmax(q_matrix)
        return q_matrix_softmax.gather(1, actions).squeeze(-1)  # [B]

    def _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        probs = self._get_states_actions_probs_impl(states, actions)
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
        return self._q_net.get_state()

    def set_state(self, policy_state: dict) -> None:
        self._q_net.set_state(policy_state)

    def soft_update(self, other_policy: RLPolicy, tau: float) -> None:
        assert isinstance(other_policy, ValueBasedPolicy)
        self._q_net.soft_update(other_policy.q_net, tau)

    def _to_device_impl(self, device: torch.device) -> None:
        self._q_net.to(device)


class DiscretePolicyGradient(DiscreteRLPolicy):
    """Policy gradient for discrete action spaces.

    Args:
        name (str): Name of the policy.
        policy_net (DiscretePolicyNet): The core net of this policy.
        trainable (bool, default=True): Whether this policy is trainable.
    """

    def __init__(
        self,
        name: str,
        policy_net: DiscretePolicyNet,
        trainable: bool = True,
    ) -> None:
        assert isinstance(policy_net, DiscretePolicyNet)

        super(DiscretePolicyGradient, self).__init__(
            name=name,
            state_dim=policy_net.state_dim,
            action_num=policy_net.action_num,
            trainable=trainable,
        )

        self._policy_net = policy_net

    @property
    def policy_net(self) -> DiscretePolicyNet:
        return self._policy_net

    def _get_actions_impl(self, states: torch.Tensor) -> torch.Tensor:
        return self._policy_net.get_actions(states, self._is_exploring)

    def _get_actions_with_probs_impl(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._policy_net.get_actions_with_probs(states, self._is_exploring)

    def _get_actions_with_logps_impl(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._policy_net.get_actions_with_logps(states, self._is_exploring)

    def _get_states_actions_probs_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self._policy_net.get_states_actions_probs(states, actions)

    def _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self._policy_net.get_states_actions_logps(states, actions)

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
        return self._policy_net.get_state()

    def set_state(self, policy_state: dict) -> None:
        self._policy_net.set_state(policy_state)

    def soft_update(self, other_policy: RLPolicy, tau: float) -> None:
        assert isinstance(other_policy, DiscretePolicyGradient)
        self._policy_net.soft_update(other_policy.policy_net, tau)

    def get_action_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Get the probabilities for all actions according to states.

        Args:
            states (torch.Tensor): States.

        Returns:
            action_probs (torch.Tensor): Action probabilities with shape [batch_size, action_num].
        """
        assert self._shape_check(
            states=states,
        ), f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."
        action_probs = self._policy_net.get_action_probs(states)
        assert match_shape(action_probs, (states.shape[0], self.action_num)), (
            f"Action probabilities shape check failed. Expecting: {(states.shape[0], self.action_num)}, "
            f"actual: {action_probs.shape}."
        )
        return action_probs

    def get_action_logps(self, states: torch.Tensor) -> torch.Tensor:
        """Get the log-probabilities for all actions according to states.

        Args:
            states (torch.Tensor): States.

        Returns:
            action_logps (torch.Tensor): Action probabilities with shape [batch_size, action_num].
        """
        return torch.log(self.get_action_probs(states))

    def _get_state_action_probs_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_probs = self.get_action_probs(states)
        return action_probs.gather(1, actions).squeeze(-1)  # [B]

    def _get_state_action_logps_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_logps = self.get_action_logps(states)
        return action_logps.gather(1, actions).squeeze(-1)  # [B]

    def _to_device_impl(self, device: torch.device) -> None:
        self._policy_net.to(device)
