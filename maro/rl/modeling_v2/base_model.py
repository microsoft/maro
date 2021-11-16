# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Optional, Tuple

import torch

from maro.rl.utils import match_shape


class AbsCoreModel(torch.nn.Module):
    """
    The ancestor of all Torch models in MARO.

    All concrete classes that inherit `AbsCoreModel` should implement all abstract methods
    declared in `AbsCoreModel`, includes:
    - step(self, loss: torch.tensor) -> None:
    - get_gradients(self, loss: torch.tensor) -> torch.tensor:
    - apply_gradients(self, grad: dict) -> None:
    - get_state(self) -> object:
    - set_state(self, state: object) -> None:
    """

    def __init__(self) -> None:
        super(AbsCoreModel, self).__init__()

    @abstractmethod
    def step(self, loss: torch.tensor) -> None:
        """Use a computed loss to back-propagate gradients and apply them to the underlying parameters.

        Args:
            loss: Result of a computation graph that involves the underlying parameters.
        """
        raise NotImplementedError

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
    """
    Simple neural network that has one input and one output.

    `SimpleNetwork` does not contain any semantics and therefore can be used for any purpose. However, we recommend
    users to use `PolicyNetwork` if the network is used for generating actions according to states. `PolicyNetwork`
    has better supports for these functionalities.

    All concrete classes that inherit `SimpleNetwork` should implement the following abstract methods:
    - Declared in `AbsCoreModel`:
        - step(self, loss: torch.tensor) -> None:
        - get_gradients(self, loss: torch.tensor) -> torch.tensor:
        - apply_gradients(self, grad: dict) -> None:
        - get_state(self) -> object:
        - set_state(self, state: object) -> None:
    - Declared in `SimpleNetwork`:
        - _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(SimpleNetwork, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim

    @property
    def input_dim(self) -> int:
        """Input dimension of the network."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Output dimension of the network."""
        return self._input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    @abstractmethod
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        The implementation that contains the actual logic of the network. Users should implement their own logics
        in this method.
        """
        pass


class ShapeCheckMixin:
    """
    Mixin that contains the `_policy_net_shape_check` method, which is used for checking whether the states and actions
    have valid shapes. Usually, it should contains three parts:
    1. Check of states' shape.
    2. Check of actions' shape.
    3. Check whether states and actions have identical batch sizes.

    `actions` is optional. If it is None, it means we do not need to check action related issues.
    """
    @abstractmethod
    def _policy_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
        pass


class PolicyNetwork(ShapeCheckMixin, AbsCoreModel):
    """
    Neural networks for policies.

    All concrete classes that inherit `PolicyNetwork` should implement the following abstract methods:
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
        super(PolicyNetwork, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim

    @property
    def state_dim(self) -> int:
        """State dimension."""
        return self._state_dim

    @property
    def action_dim(self) -> int:
        """Action dimension"""
        return self._action_dim

    def _policy_net_shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor]) -> bool:
        return all([
            states.shape[0] > 0 and match_shape(states, (None, self.state_dim)),
            actions is None or (actions.shape[0] > 0 and match_shape(actions, (None, self.action_dim))),
            actions is None or states.shape[0] == actions.shape[0]
        ])

    def get_actions(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        """
        Get actions according to the given states. The actual logics should be implemented in `_get_actions_impl`.

        Args:
            states (torch.Tensor): States with shape [batch_size, state_dim].
            exploring (bool): Get the actions under exploring mode (True) or exploiting mode (False).

        Returns:
            actions (torch.Tensor) with shape [batch_size, action_dim].
        """
        assert self._policy_net_shape_check(states=states, actions=None)
        ret = self._get_actions_impl(states, exploring)
        assert match_shape(ret, (states.shape[0], self._action_dim))
        return ret

    @abstractmethod
    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        """
        Implementation of `get_actions`.

        Args:
            states (torch.Tensor): States with shape [batch_size, state_dim].
            exploring (bool): Get the actions under exploring mode (True) or exploiting mode (False).

        Returns:
            actions (torch.Tensor) with shape [batch_size, action_dim].
        """
        pass


class DiscretePolicyNetworkMixin:
    """
    Mixin for discrete policy networks. All policy networks that generate discrete actions should extend this mixin
    and implemented all methods inherited from this mixin.
    """
    @property
    def action_num(self) -> int:
        """
        Returns the number of actions.
        """
        return self._get_action_num()

    @abstractmethod
    def _get_action_num(self) -> int:
        """
        Implementation of `action_num`.
        """
        pass


class DiscreteProbPolicyNetworkMixin(DiscretePolicyNetworkMixin, ShapeCheckMixin):
    """
    Mixin for discrete policy networks that have the concept of 'probability'. Policy networks that extend this mixin
    should first calculate the probability for each potential action, and then choose the action according to the
    probabilities.

    Notice: any concrete class that inherits `DiscreteProbPolicyNetworkMixin` should also implement
    `_get_action_num(self) -> int:` defined in `DiscretePolicyNetworkMixin`.
    """
    def get_probs(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get probabilities of all potential actions. The actual logics should be implemented in `_get_probs_impl`.

        Args:
            states (torch.Tensor): States with shape [batch_size, state_dim].

        Returns:
            probability matrix: [batch_size, action_num]
        """
        self._policy_net_shape_check(states=states, actions=None)
        ret = self._get_probs_impl(states)
        assert match_shape(ret, (states.shape[0], self.action_num))
        return ret

    @abstractmethod
    def _get_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        """
        Implementation of `get_probs`.

        Args:
            states (torch.Tensor): States with shape [batch_size, state_dim].

        Returns:
            probability matrix: [batch_size, action_num]
        """
        pass

    def get_logps(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get log-probabilities of all possible actions.

        Args:
            states (torch.Tensor): States with shape [batch_size, state_dim].

        Returns:
            Log-probability matrix: [batch_size, action_num]
        """
        return torch.log(self.get_probs(states))

    def get_actions_and_logps(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get actions and corresponding log-probabilities.

        If under exploring mode (exploring = True), the actions shall be taken follow the logic implemented in
        `_get_actions_and_logps_exploration_impl`. If under exploiting mode (exploring = False), the actions
        will be taken through a greedy strategy (choose the action with the highest probability).

        Args:
            states (torch.Tensor): States with shape [batch_size, state_dim].
            exploring (bool): `True` means under exploring mode. `False` means under exploiting mode.

        Returns:
            Actions and log-P values, both with shape [batch_size].
        """
        if exploring:
            actions, logps = self._get_actions_and_logps_exploring_impl(states)
        else:
            action_prob = self.get_logps(states)  # [batch_size, num_actions]
            logps, actions = action_prob.max(dim=1)
        assert match_shape(actions, (states.shape[0],))
        assert match_shape(logps, (states.shape[0],))
        return actions, logps

    @abstractmethod
    def _get_actions_and_logps_exploring_impl(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get actions and corresponding log-probabilities under exploring mode.

        Args:
            states (torch.Tensor): States with shape [batch_size, state_dim].

        Returns:
            Actions and log-P values, both with shape [batch_size].
        """
        pass


class ContinuousPolicyNetworkMixin:
    """
    Mixin for continuous policy networks. All policy networks that generate continuous actions should extend this mixin
    and implemented all methods inherited from this mixin.
    """
    @property
    def action_range(self) -> Tuple[float, float]:
        """
        Returns the range of actions.
        """
        return self._get_action_range()

    @abstractmethod
    def _get_action_range(self) -> Tuple[float, float]:
        """
        Implementation of `action_range`.
        """
        pass
