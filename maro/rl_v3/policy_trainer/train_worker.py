from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Union

import torch

from maro.communication import Proxy
from maro.rl.data_parallelism import TaskQueueClient
from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.utils import MultiTransitionBatch, TransitionBatch


class AbsTrainWorker(object, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        device: torch.device,
        enable_data_parallelism: bool = False
    ) -> None:
        super(AbsTrainWorker, self).__init__()
        self._name = name
        self._enable_data_parallelism = enable_data_parallelism
        self._task_queue_client: Optional[TaskQueueClient] = None
        self._device = device

    @property
    def name(self) -> str:
        return self._name

    def _get_batch_grad(
        self,
        batch: Union[TransitionBatch, MultiTransitionBatch],
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
        if self._enable_data_parallelism:
            assert self._task_queue_client is not None
            raise NotImplementedError  # TODO
        else:
            return self.get_batch_grad(batch, tensor_dict, scope)

    @abstractmethod
    def get_batch_grad(
        self,
        batch: Union[TransitionBatch, MultiTransitionBatch],
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
        raise NotImplementedError

    def init_data_parallel(self, *args, **kwargs) -> None:
        """
        Initialize a proxy in the policy, for data-parallel training.
        Using the same arguments as `Proxy`.
        """
        self._task_queue_client = TaskQueueClient()
        self._task_queue_client.create_proxy(*args, **kwargs)

    def init_data_parallel_with_existing_proxy(self, proxy: Proxy) -> None:
        """
        Initialize a proxy in the policy with an existing one, for data-parallel training.
        """
        self._task_queue_client = TaskQueueClient()
        self._task_queue_client.set_proxy(proxy)

    def exit_data_parallel(self) -> None:
        if self._task_queue_client is not None:
            self._task_queue_client.exit()
            self._task_queue_client = None

    @abstractmethod
    def get_worker_state_dict(self, scope: str = "all") -> dict:
        """
        Returns:
            A dict that contains worker's state.
        """
        raise NotImplementedError

    @abstractmethod
    def set_worker_state_dict(self, worker_state_dict: dict, scope: str = "all") -> None:
        """Set trainer's state."""
        raise NotImplementedError


class SingleTrainWorker(AbsTrainWorker, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        device: torch.device,
        enable_data_parallelism: bool = False
    ) -> None:
        super(SingleTrainWorker, self).__init__(name, device, enable_data_parallelism)
        self._batch: Optional[TransitionBatch] = None
        self._policy: Optional[RLPolicy] = None

    def register_policy(self, policy: RLPolicy) -> None:
        self._register_policy_impl(policy)

    @abstractmethod
    def _register_policy_impl(self, policy: RLPolicy) -> None:
        raise NotImplementedError

    def set_batch(self, batch: TransitionBatch) -> None:
        self._batch = batch

    def get_policy_state(self) -> object:
        return self._policy.get_policy_state()

    def set_policy_state(self, policy_state: object) -> None:
        self._policy.set_policy_state(policy_state)


class MultiTrainWorker(AbsTrainWorker, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        device: torch.device,
        enable_data_parallelism: bool = False
    ) -> None:
        super(MultiTrainWorker, self).__init__(name, device, enable_data_parallelism)
        self._batch: Optional[MultiTransitionBatch] = None
        self._policies: Optional[Dict[int, RLPolicy]] = None
        self._indexes: List[int] = []

    @property
    def num_policies(self) -> int:
        return len(self._policies)

    def register_policies(self, policy_dict: Dict[int, RLPolicy]) -> None:
        self._indexes = list(policy_dict.keys())
        self._register_policies_impl(policy_dict)

    @abstractmethod
    def _register_policies_impl(self, policy_dict: Dict[int, RLPolicy]) -> None:
        raise NotImplementedError

    def set_batch(self, batch: MultiTransitionBatch) -> None:
        self._batch = batch

    def get_policy_state_dict(self) -> dict:
        return {i: policy.get_policy_state() for i, policy in self._policies.items()}

    def set_policy_state_dict(self, policy_state_dict: dict) -> None:
        for i, policy in self._policies.items():
            policy.set_policy_state(policy_state_dict[i])