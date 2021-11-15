from abc import ABCMeta, abstractmethod
from typing import Iterable, List, Optional, Union

import numpy as np
import torch

from maro.communication import Proxy
from maro.rl.data_parallelism.task_queue import TaskQueueClient
from maro.rl.policy_v2.policy_interfaces import ShapeCheckMixin
from maro.rl.utils import match_shape


class AbsPolicy(object):
    """Abstract policy class.

    All concrete classes that inherit `AbsPolicy` should implement the following abstract methods:
    - __call__(self, states: object) -> object:
    - _get_state_dim(self) -> int:
    """

    def __init__(self, name: str) -> None:
        """

        Args:
            name (str): Unique identifier for the policy.
        """
        super().__init__()
        print(f"Initializing {self.__class__.__module__}.{self.__class__.__name__}")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def __call__(self, states: object) -> object:
        """Get actions and other auxiliary information based on states.

        Args:
            states (object): environment states.

        Returns:
            Actions and other auxiliary information based on states.
            The format of the returns is defined by the policy.
        """
        raise NotImplementedError

    @property
    def state_dim(self) -> int:
        return self._get_state_dim()

    @abstractmethod
    def _get_state_dim(self) -> int:
        raise NotImplementedError


class DummyPolicy(AbsPolicy):
    """Dummy policy that does nothing.

    Note that the meaning of a "None" action may depend on the scenario.
    """

    def __init__(self, name: str) -> None:
        super(DummyPolicy, self).__init__(name)

    def __call__(self, states: object) -> object:
        return None

    def _get_state_dim(self) -> int:
        return -1


class RuleBasedPolicy(AbsPolicy):
    """
    Rule-based policy that generates actions according to a fixed rule.
    The rule is immutable, which means a rule-based policy is not trainable.

    All concrete classes that inherit `RuleBasedPolicy` should implement the following abstract methods:
    - Declared in `AbsPolicy`:
        - _get_state_dim(self) -> int:
    - Declared in `RuleBasedPolicy`:
        - _rule(self, state: object) -> object:
    """

    def __init__(self, name: str) -> None:
        super(RuleBasedPolicy, self).__init__(name)

    def __call__(self, states: object) -> object:
        return self._rule(states)

    @abstractmethod
    def _rule(self, state: object) -> object:
        """The rule that should be implemented by inheritors."""
        raise NotImplementedError


class AbsRLPolicy(ShapeCheckMixin, AbsPolicy):
    """Policy that learns from simulation experiences.
    Reinforcement learning (RL) policies should inherit from this.

    All concrete classes that inherit `AbsRLPolicy` should implement the following abstract methods:
    - Declared in `AbsPolicy`:
        - _get_state_dim(self) -> int:
    - Declared in `AbsRLPolicy`:
        - _call_impl(self, states: np.ndarray) -> Iterable:
        - record(self, ...) -> None:
        - get_rollout_info(self) -> object:
        - get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> object:
        - data_parallel(self, *args, **kwargs) -> None:
        - learn_with_data_parallel(self, batch: dict) -> None:
        - update(self, loss_info_list: List[dict]) -> None:
        - learn(self, batch: dict) -> None:
        - improve(self) -> None:
        - get_state(self) -> object:
        - set_state(self, policy_state: object) -> None:
        - load(self, path: str) -> None:
        - save(self, path: str) -> None:
    - Declared in `ShapeCheckMixin`:
        - _shape_check(self, states: np.ndarray, actions: Optional[np.ndarray]) -> bool:
    """
    def __init__(self, name: str, device: str) -> None:
        """
        Args:
            name (str): Name of the policy.
            device (str): Device that uses to train the Torch model.
        """
        super(AbsRLPolicy, self).__init__(name)
        self._exploration_params = {}
        self._exploring = True
        self._proxy = Optional[Proxy]

        self._device = torch.device(device) if device is not None \
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def exploration_params(self) -> dict:
        return self._exploration_params

    def explore(self) -> None:
        """Switch the policy to the exploring mode."""
        self._exploring = True

    def exploit(self) -> None:
        """Switch the policy to the exploiting mode."""
        self._exploring = False

    def get_exploration_params(self):
        raise NotImplementedError

    def exploration_step(self):
        raise NotImplementedError

    def ndarray_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(self._device)

    def __call__(self, states: np.ndarray) -> Iterable:
        assert self._shape_check(states=states, actions=None)
        ret = self._call_impl(states)
        assert self._call_post_check(states=states, ret=ret)
        return ret

    @abstractmethod
    def _call_impl(self, states: np.ndarray) -> Iterable:
        """The implementation of `__call__` method. Actual logic should be implemented under this method."""
        raise NotImplementedError

    @abstractmethod
    def _call_post_check(self, states: np.ndarray, ret: Iterable) -> bool:
        raise NotImplementedError

    @abstractmethod
    def record(
        self,
        agent_id: str,
        state: np.ndarray,
        action: object,
        reward: float,
        next_state: np.ndarray,
        terminal: bool
    ) -> None:
        """Record a transition in an internal buffer or memory.

        Since we may have multiple agents sharing this policy, the internal buffer / memory should use the agents'
        names to separate storage for these agents. The ``agent_id`` parameter serves this purpose.
        """
        raise NotImplementedError

    @abstractmethod
    def get_rollout_info(self) -> object:  # TODO: return type?
        """Extract information from the recorded transitions.

        Implement this method if you are doing distributed learning. What this function returns will be used to update
        policy parameters on the learning side (abstracted through ``AbsPolicyManager``) with or without invoking the
        policy improvement algorithm, depending on the type of information. If you want the policy improvement algorithm
        to be invoked on roll-out instances (i.e., in distributed fashion), this should return loss information (which
        can be obtained by calling ``get_batch_loss`` function with ``explicit_grad`` set to True) to be used by
        ``update`` on the learning side. If you want the policy improvement algorithm to be invoked on the learning
        side, this should return a data batch to be used by ``learn`` on the learning side. See the implementation of
        this function in ``ActorCritic`` for reference.
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> object:  # TODO: return type?
        """Compute policy improvement information, i.e., loss, from a data batch.

        This can be used as a sub-routine in ``learn`` and ``improve``, as these methods usually require computing
        loss from a batch.

        Args:
            batch (dict): Data batch to compute the policy improvement information for.
            explicit_grad (bool): If True, the gradients should be explicitly returned. Defaults to False.
        """
        raise NotImplementedError

    def data_parallel(self, *args, **kwargs) -> None:
        """"Initialize a proxy in the policy, for data-parallel training.
        Using the same arguments as `Proxy`."""
        self.task_queue_client = TaskQueueClient()
        self.task_queue_client.create_proxy(*args, **kwargs)

    def data_parallel_with_existing_proxy(self, proxy: Proxy) -> None:
        """"Initialize a proxy in the policy with an existing one, for data-parallel training."""
        self.task_queue_client = TaskQueueClient()
        self.task_queue_client.set_proxy(proxy)

    def exit_data_parallel(self) -> None:
        if hasattr(self, "task_queue_client"):
            self.task_queue_client.exit()

    @abstractmethod
    def learn_with_data_parallel(self, batch: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, loss_info_list: List[dict]) -> None:
        """Update with loss information computed by multiple sources.

        There are two possible scenarios where you need to implement this interface: 1) if you are doing distributed
        learning and want each roll-out instance to collect information that can be used to update policy parameters
        on the learning side (abstracted through ``AbsPolicyManager``) without invoking the policy improvement
        algorithm. Such information usually includes gradients with respect to the policy parameters. An example where
        this can be useful is the Asynchronous Advantage Actor Critic (A3C) (https://arxiv.org/abs/1602.01783);
        2) if you are computing loss in data-parallel fashion, i.e., by splitting a data batch to several smaller
        batches and sending them to a set of remote workers for parallelized loss computation.

        Args:
            loss_info_list (List[dict]): A list of dictionaries containing loss information (e.g., gradients) computed
                by multiple sources.
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self, batch: dict) -> None:
        """Learn from a batch of roll-out data.

        Implement this interface if you are doing distributed learning and want the roll-out instances to collect
        information that can be used to update policy parameters on the learning side (abstracted through
        ``AbsPolicyManager``) using the policy improvement algorithm.

        Args:
            batch (dict): Training data to train the policy with.
        """
        raise NotImplementedError

    @abstractmethod
    def improve(self) -> None:
        """Learn using data collected locally.

        Implement this interface if you are doing single-threaded learning where a single policy instance is used for
        roll-out and training. The policy should have some kind of internal buffer / memory to store roll-out data and
        use as the source of training data.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> object:
        """Return the current state of the policy.

        The implementation must be in correspondence with that of ``set_state``. For example, if a torch model
        is contained in the policy, ``get_state`` may include a call to ``state_dict()`` on the model, while
        ``set_state`` should accordingly include ``load_state_dict()``.
        """
        raise NotImplementedError

    @abstractmethod
    def set_state(self, policy_state: object) -> None:
        """Set the policy state to ``policy_state``.

        The implementation must be in correspondence with that of ``get_state``. For example, if a torch model
        is contained in the policy, ``set_state`` may include a call to ``load_state_dict()`` on the model, while
        ``get_state`` should accordingly include ``state_dict()``.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the policy state from disk."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the policy state to disk."""
        raise NotImplementedError


class SingleRLPolicy(AbsRLPolicy, metaclass=ABCMeta):
    """Single-agent policy that learns from simulation experiences.

    All concrete classes that inherit `SingleRLPolicy` should implement the following abstract methods:
    - Declared in `AbsPolicy`:
        - _get_state_dim(self) -> int:
    - Declared in `RLPolicy`:
        - _call_impl(self, states: np.ndarray) -> Iterable:
        - record(self, ...) -> None:
        - get_rollout_info(self) -> object:
        - get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> object:
        - data_parallel(self, *args, **kwargs) -> None:
        - learn_with_data_parallel(self, batch: dict) -> None:
        - update(self, loss_info_list: List[dict]) -> None:
        - learn(self, batch: dict) -> None:
        - improve(self) -> None:
        - get_state(self) -> object:
        - set_state(self, policy_state: object) -> None:
        - load(self, path: str) -> None:
        - save(self, path: str) -> None:
    - Declared in `SingleRLPolicy`:
        - _get_action_dim(self) -> int:
        - _get_actions_impl(self, states: np.ndarray) -> np.ndarray:
    """
    def __init__(self, name: str, device: str) -> None:
        """
        Args:
            name (str): Name of the policy.
            device (str): Device that uses to train the Torch model.
        """
        super(SingleRLPolicy, self).__init__(name=name, device=device)

    def _shape_check(self, states: np.ndarray, actions: Union[None, np.ndarray]) -> bool:
        return all([
            states.shape[0] > 0 and match_shape(states, (None, self.state_dim)),
            actions is None or (actions.shape[0] > 0 and match_shape(actions, (None, self.action_dim))),
            actions is None or states.shape[0] == actions.shape[0]
        ])

    @property
    def action_dim(self) -> int:
        return self._get_action_dim()

    @abstractmethod
    def _get_action_dim(self) -> int:
        raise NotImplementedError

    def get_actions(self, states: np.ndarray) -> np.ndarray:
        assert self._shape_check(states=states, actions=None)
        actions = self._get_actions_impl(states)
        assert self._shape_check(states=states, actions=actions)
        return actions

    @abstractmethod
    def _get_actions_impl(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MultiRLPolicy(AbsRLPolicy, metaclass=ABCMeta):
    """Multi-agent policy that learns from simulation experiences.

    All concrete classes that inherit `MultiRLPolicy` should implement the following abstract methods:
    - Declared in `AbsPolicy`:
        - _get_state_dim(self) -> int:
    - Declared in `RLPolicy`:
        - _call_impl(self, states: np.ndarray) -> Iterable:
        - record(self, ...) -> None:
        - get_rollout_info(self) -> object:
        - get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> object:
        - data_parallel(self, *args, **kwargs) -> None:
        - learn_with_data_parallel(self, batch: dict) -> None:
        - update(self, loss_info_list: List[dict]) -> None:
        - learn(self, batch: dict) -> None:
        - improve(self) -> None:
        - get_state(self) -> object:
        - set_state(self, policy_state: object) -> None:
        - load(self, path: str) -> None:
        - save(self, path: str) -> None:
    - Declared in `MultiRLPolicy`:
        - _get_action_dims(self) -> List[int]:
        - _get_agent_num(self) -> int:
    """
    def __init__(self, name: str, device: str) -> None:
        """
        Args:
            name (str): Name of the policy.
            device (str): Device that uses to train the Torch model.
        """
        super(MultiRLPolicy, self).__init__(name=name, device=device)

    def __call__(self, states: List[np.ndarray], agent_ids: List[int]):
        assert self._shape_check(states=states, actions=None)
        ret = self._call_impl(states, agent_ids)
        # assert self._call_post_check(states=states, ret=ret)  # TODO
        return ret

    def _shape_check(self, states: List[np.ndarray], actions: Union[None, List[torch.Tensor]]) -> bool:
        for state in states:
            if not state.shape[0] > 0:
                return False

        if actions is not None:
            for action, action_dim in zip(actions, self.action_dims):
                if not match_shape(action, (action_dim, )):
                    return False

        return True

    @property
    def action_dims(self) -> List[int]:
        return self._get_action_dims()

    @abstractmethod
    def _get_action_dims(self) -> List[int]:
        raise NotImplementedError

    @property
    def agent_num(self) -> int:
        return self._get_agent_num()

    @abstractmethod
    def _get_agent_num(self) -> int:
        raise NotImplementedError

    def get_actions(self, states: Union[np.ndarray, List[np.ndarray]], agent_ids: List[int]) -> List[np.ndarray]:
        assert self._shape_check(states=states, actions=None)
        actions = self._get_actions_impl(states, agent_ids)
        assert self._shape_check(states=states, actions=actions)
        return actions

    @abstractmethod
    def _get_actions_impl(self, states: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        raise NotImplementedError
