# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List

import torch

from maro.rl.utils import average_grads
from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.utils import AbsTransitionBatch, MultiTransitionBatch, TransitionBatch


class AbsTrainOps(object, metaclass=ABCMeta):
    """The basic component for training a policy, which takes charge of gradient computation and policy update.
    Each ops is used for training a single policy. An ops is an atomic unit in the distributed mode.
    """
    def __init__(
        self,
        device: str,
        is_single_scenario: bool,
        get_policy_func: Callable[[], RLPolicy],
        enable_data_parallelism: bool = False,
    ) -> None:
        """
        Args:
            device (str): Identifier for the torch device. The policy will be moved to the specified device. If it is
                None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise. Defaults to None.
            is_single_scenario (bool): Identifier of whether this ops is used under a single trainer or a multi trainer.
            get_policy_func (Callable[[], RLPolicy]): Function used to create the policy of this ops.
            enable_data_parallelism (bool): Whether to enable data parallelism in this trainer. Defaults to False.
        """
        super(AbsTrainOps, self).__init__()
        self._device = torch.device(device) if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_single_scenario = is_single_scenario
        self._enable_data_parallelism = enable_data_parallelism

        # Create the policy and put it on the right device.
        if self._is_single_scenario:
            self._policy = get_policy_func()
            self._policy.to_device(self._device)

    @property
    def policy_name(self) -> str:
        return self._policy.name

    def policy_state_dim(self) -> int:
        return self._policy.state_dim

    def policy_action_dim(self) -> int:
        return self._policy.action_dim

    def _is_valid_transition_batch(self, batch: AbsTransitionBatch) -> bool:
        """Used to check the transition batch's type. If this ops is used under a single trainer, the batch should be
        a `TransitionBatch`. Otherwise, it should be a `MultiTransitionBatch`.
        """
        return isinstance(batch, TransitionBatch) if self._is_single_scenario \
            else isinstance(batch, MultiTransitionBatch)

    def _get_batch_grad(
        self,
        batch: AbsTransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Calculate the gradients of the given batch, with the auxiliary tensors.

        Args:
            batch (AbsTransitionBatch): The training batch.
            tensor_dict (Dict[str, object]): Auxiliary tensors used in the calculation. Defaults to None.
            scope (str): The scope of the parts that should be calculated. Defaults to 'all'.

        Returns:
            A dict with format: {part identifier: {param name: gradient}}
        """
        if self._enable_data_parallelism:
            gradients = self._remote_learn(batch, tensor_dict, scope)
            return average_grads(gradients)
        else:
            return self.get_batch_grad(batch, tensor_dict, scope)

    def _remote_learn(
        self,
        batch: AbsTransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> List[Dict[str, Dict[int, Dict[str, torch.Tensor]]]]:
        """Learn a batch of experience data from remote gradient workers.
        The task queue client will first request available gradient workers from task queue. If all workers are busy,
        it will keep waiting until at least 1 worker is available. Then the task queue client submits batch and state
        to the assigned workers to compute gradients.
        """
        pass  # TODO

    @abstractmethod
    def get_batch_grad(
        self,
        batch: AbsTransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """The actual logic of gradients calculation.

        Args:
            batch (AbsTransitionBatch): The training batch.
            tensor_dict (Dict[str, object]): Auxiliary tensors used in the calculation. Defaults to None.
            scope (str): The scope of the parts that should be calculated. Defaults to 'all'.

        Returns:
            A dict with format: {part identifier: {param name: gradient}}
        """
        raise NotImplementedError

    @abstractmethod
    def _dispatch_batch(self, batch: AbsTransitionBatch, num_sub_batches: int) -> List[AbsTransitionBatch]:
        """Divide experience data batch to several parts.
        For on-policy algorithms, like PG, the batch is divided into several complete trajectories.
        For off-policy algorithms, like DQN, the batch is treated as independent data points and divided evenly."""
        raise NotImplementedError

    @abstractmethod
    def _dispatch_tensor_dict(self, tensor_dict: Dict[str, object], num_sub_batches: int) -> List[Dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def get_state_dict(self, scope: str = "all") -> dict:
        """
        Returns:
            A dict that contains ops's state.
        """
        raise NotImplementedError

    @abstractmethod
    def set_state_dict(self, ops_state_dict: dict, scope: str = "all") -> None:
        """Set ops's state."""
        raise NotImplementedError

    def set_batch(self, batch: AbsTransitionBatch) -> None:
        assert self._is_valid_transition_batch(batch)
        self._batch = batch

    def get_policy_state(self) -> object:
        return self._policy.name, self._policy.get_state()

    def set_policy_state(self, policy_state: object) -> None:
        self._policy.set_state(policy_state)
