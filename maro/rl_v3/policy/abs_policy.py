from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from maro.rl_v3.utils import ActionWithAux, match_shape
from maro.rl_v3.utils.objects import SHAPE_CHECK_FLAG


class AbsPolicy(object, metaclass=ABCMeta):
    """
    Policy. A policy takes states as inputs and generates actions as outputs. A policy cannot update itself. It has to
    be updated by external trainers through public interfaces.
    """

    def __init__(self, name: str, trainable: bool) -> None:
        """
        Args:
            name (str): Name of this policy.
            trainable (bool): Whether this policy is trainable.
        """
        super(AbsPolicy, self).__init__()

        self._name = name
        self._trainable = trainable

    @abstractmethod
    def get_actions(self, states: object) -> object:
        """
        Get actions according to states.

        Args:
            states (object): States.

        Returns:
            Actions.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @property
    def trainable(self) -> bool:
        return self._trainable


class DummyPolicy(AbsPolicy):
    """
    Dummy policy that takes no actions.
    """
    def __init__(self) -> None:
        super(DummyPolicy, self).__init__(name='DUMMY_POLICY', trainable=False)

    def get_actions(self, states: object) -> object:
        return None


class RuleBasedPolicy(AbsPolicy, metaclass=ABCMeta):
    """
    Rule-based policy. The user should implement the rule of this policy, and a rule-based policy is not trainable.
    """
    def __init__(self, name: str) -> None:
        super(RuleBasedPolicy, self).__init__(name=name, trainable=False)

    def get_actions(self, states: object) -> object:
        return self._rule(states)

    @abstractmethod
    def _rule(self, states: object) -> object:
        raise NotImplementedError


class RLPolicy(AbsPolicy, metaclass=ABCMeta):
    """
    Reinforcement learning policy.
    """
    def __init__(
        self,
        name: str,
        state_dim: int,
        action_dim: int,
        device: str = None,
        trainable: bool = True
    ) -> None:
        """
        Args:
            name (str): Name of the policy.
            state_dim (int): Dimension of states.
            action_dim (int): Dimension of actions.
            device (str): Device to store this model ('cpu' or 'gpu').
            trainable (bool): Whether this policy is trainable. Defaults to True.
        """
        super(RLPolicy, self).__init__(name=name, trainable=trainable)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._device = torch.device(device) if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_exploring = False

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def is_exploring(self) -> bool:
        """
        Whether this policy is under exploring mode.
        """
        return self._is_exploring

    def explore(self) -> None:
        """
        Set the policy to exploring mode.
        """
        self._is_exploring = True

    def exploit(self) -> None:
        """
        Set the policy to exploiting mode.
        """
        self._is_exploring = False

    def ndarray_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """
        Convert a np.ndarray to a torch.Tensor.

        Args:
            array (np.ndarray): The input ndarray.

        Returns:
            A tensor with same shape and values.
        """
        return torch.from_numpy(array).to(self._device)

    @abstractmethod  # TODO
    def step(self, loss: torch.Tensor) -> None:
        """
        Run a training step to update the policy.

        Args:
            loss (torch.Tensor): Loss used to update the policy.
        """
        raise NotImplementedError

    @abstractmethod  # TODO
    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get the gradients with respect to all parameters of the internal nets according to the given loss.

        Args:
            loss (torch.tensor): Loss used to update the model.

        Returns:
            A dict that contains gradients of the internal nets for all parameters.
        """
        raise NotImplementedError

    def get_actions_with_aux(self, states: np.ndarray) -> List[ActionWithAux]:
        """
        Get the action with optional auxiliary information according to states.

        Args:
            states (np.ndarray): States.

        Returns:
            Actions with optional auxiliary information (ActionWithAux).
        """
        actions, logps = self.get_actions_with_logps(states, require_logps=True)
        values = self.get_values_by_states_and_actions(states, actions)

        size = len(actions)
        actions_with_aux = []
        for i in range(size):
            actions_with_aux.append(ActionWithAux(
                action=actions[i],
                value=values[i] if values is not None else None,
                logp=logps[i] if logps is not None else None
            ))
        return actions_with_aux

    @abstractmethod
    def get_values_by_states_and_actions(self, states: np.ndarray, actions: np.ndarray) -> Optional[np.ndarray]:
        """
        Get the state values according to states and actions. This method is meaningful only for value-base policies.
        For policy gradient policies, just return None.

        Args:
            states (np.ndarray): States.
            actions (np.ndarray): Actions.

        Returns:
            State values with shape [batch_size] or None.
        """
        raise NotImplementedError

    def get_actions(self, states: np.ndarray) -> np.ndarray:
        return self.get_actions_with_logps(states, require_logps=False)[0]

    def get_actions_tensor(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get actions according to states. Takes torch.Tensor as inputs and returns torch.Tensor.

        Args:
            states (torch.Tensor): States.

        Returns:
            Actions, a torch.Tensor.
        """
        return self.get_actions_with_logps_tensor(states, require_logps=False)[0]

    @abstractmethod
    def _get_actions_with_logps_impl(
        self, states: torch.Tensor, exploring: bool, require_logps: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Implementation of `get_actions_with_logps_tensor`.
        """
        raise NotImplementedError

    def get_actions_with_logps(
        self, states: np.ndarray, require_logps: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get actions, and log probabilities of the actions if required, according to the states.

        Args:
            states (torch.Tensor): States.
            require_logps (bool): If the return value should contains log probabilities. Defaults to True.

        Returns:
            Actions
            Log probabilities if require_logps == True else None.
        """
        actions, logps = self.get_actions_with_logps_tensor(self.ndarray_to_tensor(states), require_logps)
        return actions.cpu().numpy(), logps.cpu().numpy() if logps is not None else None

    def get_actions_with_logps_tensor(
        self, states: torch.Tensor, require_logps: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Similar to `get_actions_with_logps`, but takes torch.Tensor as inputs and returns torch.Tensor.
        """
        assert self._shape_check(states=states), \
            f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."
        actions, logps = self._get_actions_with_logps_impl(states, self._is_exploring, require_logps)
        assert self._shape_check(states=states, actions=actions), \
            f"Actions shape check failed. Expecting: {(states.shape[0], self.action_dim)}, actual: {actions.shape}."
        assert logps is None or match_shape(logps, (states.shape[0],)), \
            f"Log probabilities shape check failed. Expecting: {(states.shape[0],)}, actual: {logps.shape}."
        if SHAPE_CHECK_FLAG:
            assert self._post_check(states=states, actions=actions)
        return actions, logps

    @abstractmethod
    def freeze(self) -> None:
        """
        (Partially) freeze the current model. The users should write their own strategy to determine the list of
        parameters to freeze.
        """
        raise NotImplementedError

    @abstractmethod
    def unfreeze(self) -> None:
        """
        (Partially) unfreeze the current model. The users should write their own strategy to determine the list of
        parameters to unfreeze.
        """
        raise NotImplementedError

    @abstractmethod
    def eval(self) -> None:
        """
        Switch the policy to evaluating mode.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        """
        Switch the policy to training mode.
        """
        raise NotImplementedError

    @abstractmethod
    def get_policy_state(self) -> object:
        """
        Get the state of the policy.
        """
        raise NotImplementedError

    @abstractmethod
    def set_policy_state(self, policy_state: object) -> None:
        """
        Set the state of the policy.
        """
        raise NotImplementedError

    @abstractmethod
    def soft_update(self, other_policy: RLPolicy, tau: float) -> None:
        """
        Soft update the policy's parameters according to another policy.

        Args:
            other_policy (AbsNet): The source policy. Must has same type with the current policy.
            tau (float): Soft update coefficient.
        """
        raise NotImplementedError

    def _shape_check(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> bool:
        if not SHAPE_CHECK_FLAG:
            return True
        else:
            if states.shape[0] == 0:
                return False
            if not match_shape(states, (None, self.state_dim)):
                return False

            if actions is not None:
                if not match_shape(actions, (states.shape[0], self.action_dim)):
                    return False
            return True

    @abstractmethod
    def _post_check(self, states: torch.Tensor, actions: torch.Tensor) -> bool:
        raise NotImplementedError


if __name__ == '__main__':
    data = [AbsPolicy('Jack', True), AbsPolicy('Tom', True), DummyPolicy(), DummyPolicy()]
    for policy in data:
        print(policy.name)
