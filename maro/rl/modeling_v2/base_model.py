# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Optional, Tuple

import torch

from maro.rl.utils import match_shape


class AbsCoreModel(torch.nn.Module):
    """TODO
    """
    def __init__(self):
        super(AbsCoreModel, self).__init__()

    @abstractmethod
    def step(self, loss: torch.tensor) -> None:
        """Use a computed loss to back-propagate gradients and apply them to the underlying parameters.

        Args:
            loss: Result of a computation graph that involves the underlying parameters.
        """
        pass

    @abstractmethod
    def get_gradients(self, loss: torch.tensor) -> torch.tensor:
        """Get gradients from a computed loss.

        There are two possible scenarios where you need to implement this interface: 1) if you are doing distributed
        learning and want each roll-out instance to collect gradients that can be directly applied to policy parameters
        on the learning side (abstracted through ``AbsPolicyManager``); 2) if you are computing loss in data-parallel
        fashion, i.e., by splitting a data batch to several smaller batches and sending them to a set of remote workers
        for parallelized gradient computation. In this case, this method will be used by the remote workers.
        """
        pass

    @abstractmethod
    def apply_gradients(self, grad: dict) -> None:
        """Apply gradients to the model parameters.

        This needs to be implemented together with ``get_gradients``.
        """
        pass

    @abstractmethod
    def get_state(self) -> object:
        """Return the current model state.

        Ths model state usually involves the "state_dict" of the module as well as those of the embedded optimizers.
        """
        pass

    @abstractmethod
    def set_state(self, state: object) -> None:
        """Set model state.

        Args:
            state: Model state to be applied to the instance. Ths model state is either the result of a previous call
            to ``get_state`` or something loaded from disk and involves the "state_dict" of the module as well as those
            of the embedded optimizers.
        """
        pass

    def soft_update(self, other_model: torch.nn.Module, tau: float) -> None:
        """Soft-update model parameters using another model.

        Update formulae: param = (1 - tau) * param + tau * other_param.

        Args:
            other_model: The model to update the current model with.
            tau (float): Soft-update coefficient.
        """
        for params, other_params in zip(self.parameters(), other_model.parameters()):
            params.data = (1 - tau) * params.data + tau * other_params.data


class SimpleNetwork(AbsCoreModel):
    """Simple neural network that has one input and one output.
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(SimpleNetwork, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    @abstractmethod
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ShapeCheckMixin:
    @abstractmethod
    def _policy_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
        pass


class PolicyNetwork(ShapeCheckMixin, AbsCoreModel):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(PolicyNetwork, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _policy_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
        return all([
            match_shape(states, (None, self.state_dim)),
            actions is None or match_shape(actions, (None, self.action_dim)),
            actions is None or states.shape[0] == actions.shape[0]
        ])

    def get_actions(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        assert self._is_valid_state_shape(states)
        ret = self._get_actions_impl(states, exploring)
        assert match_shape(ret, (states.shape[0], self._action_dim))
        return ret

    @abstractmethod
    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        pass


class DiscretePolicyNetworkMixin:
    @property
    def action_num(self) -> int:
        return self._get_action_num()

    @abstractmethod
    def _get_action_num(self) -> int:
        pass


class DiscreteProbPolicyNetworkMixin(DiscretePolicyNetworkMixin, ShapeCheckMixin):
    def get_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Get probabilities of all possible actions.

        Args:
            states: [batch_size, state_dim]

        Returns:
            probability matrix: [batch_size, action_num]
        """
        self._policy_net_shape_check(states, None)
        ret = self._get_probs_impl(states)
        assert match_shape(ret, (states.shape[0], self.action_num))
        return ret

    @abstractmethod
    def _get_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        pass

    def get_logps(self, states: torch.Tensor) -> torch.Tensor:
        """Get log-probabilities of all possible actions.

        Args:
            states: [batch_size, state_dim]

        Returns:
            Log-probability matrix: [batch_size, action_num]
        """
        return torch.log(self.get_probs(states))

    def get_actions_and_logps(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get actions and corresponding log-probabilities under exploration mode.

        Args:
            states: [batch_size, state_dim]
            exploring: `True` means under exploring mode. `False` means under exploiting mode.

        Returns:
            Actions and log-P values, both with shape [batch_size].
        """
        if exploring:
            actions, logps = self._get_actions_and_logps_exploration_impl(states)
            assert match_shape(actions, (states.shape[0],))
            assert match_shape(logps, (states.shape[0],))
            return actions, logps
        else:
            action_prob = self.get_logps(states)  # [batch_size, num_actions]
            logps, action = action_prob.max(dim=1)
            return action, logps

    @abstractmethod
    def _get_actions_and_logps_exploration_impl(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
