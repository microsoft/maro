from abc import abstractmethod
from typing import Iterable, List, Optional

import numpy as np
import torch

from maro.communication import Proxy
from maro.rl.policy_v2.policy_interfaces import ShapeCheckMixin
from maro.rl.utils import match_shape


class AbsPolicy(object):
    """Abstract policy class.

    Args:
        name (str): Unique identifier for the policy.

    """

    def __init__(self, name: str) -> None:
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
        pass

    @property
    def state_dim(self) -> int:
        return self._get_state_dim()

    @abstractmethod
    def _get_state_dim(self) -> int:
        pass


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
    """

    def __init__(self, name: str) -> None:
        super(RuleBasedPolicy, self).__init__(name)

    def __call__(self, states: object) -> object:
        return self._rule(states)

    @abstractmethod
    def _rule(self, state: object) -> object:
        """The rule that should be implemented by inheritors."""
        pass


class RLPolicy(ShapeCheckMixin, AbsPolicy):
    """Policy that learns from simulation experiences.

    Reinforcement learning (RL) policies should inherit from this.

    Args:
        name (str): Name of the policy.
        device (str): Device that uses to train the Torch model.
    """
    def __init__(self, name: str, device: str) -> None:
        super(RLPolicy, self).__init__(name)
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

    def __call__(self, states: np.ndarray) -> Iterable:
        assert self._shape_check(states, None)
        return self._call_impl(states)

    @abstractmethod
    def _call_impl(self, states: np.ndarray) -> Iterable:
        """The implementation of `__call__` method. Actual logic should be implemented under this method."""
        pass

    def _shape_check(self, states: np.ndarray, actions: Optional[np.ndarray]) -> bool:
        return all([
            states.shape[0] > 0 and match_shape(states, (None, self.state_dim)),
            actions is None or (actions.shape[0] > 0 and match_shape(actions, (None, 1))),
            actions is None or states.shape[0] == actions.shape[0]
        ])

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
        pass

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
        pass

    @abstractmethod
    def get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> object:  # TODO: return type?
        """Compute policy improvement information, i.e., loss, from a data batch.

        This can be used as a sub-routine in ``learn`` and ``improve``, as these methods usually require computing
        loss from a batch.

        Args:
            batch (dict): Data batch to compute the policy improvement information for.
            explicit_grad (bool): If True, the gradients should be explicitly returned. Defaults to False.
        """
        pass

    @abstractmethod
    def data_parallel(self, *args, **kwargs) -> None:
        """"Initialize a proxy in the policy, for data-parallel training.
        Using the same arguments as `Proxy`."""
        pass

    def data_parallel_with_existing_proxy(self, proxy: Proxy) -> None:
        """"Initialize a proxy in the policy with an existing one, for data-parallel training."""
        self._proxy = proxy

    def exit_data_parallel(self) -> None:
        if self._proxy is not None:
            self._proxy.close()

    @abstractmethod
    def learn_with_data_parallel(self, batch: dict, worker_id_list: list) -> None:
        pass

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
        pass

    @abstractmethod
    def learn(self, batch: dict) -> None:
        """Learn from a batch of roll-out data.

        Implement this interface if you are doing distributed learning and want the roll-out instances to collect
        information that can be used to update policy parameters on the learning side (abstracted through
        ``AbsPolicyManager``) using the policy improvement algorithm.

        Args:
            batch (dict): Training data to train the policy with.
        """
        pass

    @abstractmethod
    def improve(self) -> None:
        """Learn using data collected locally.

        Implement this interface if you are doing single-threaded learning where a single policy instance is used for
        roll-out and training. The policy should have some kind of internal buffer / memory to store roll-out data and
        use as the source of training data.
        """
        pass

    @abstractmethod
    def get_state(self) -> object:
        """Return the current state of the policy.

        The implementation must be in correspondence with that of ``set_state``. For example, if a torch model
        is contained in the policy, ``get_state`` may include a call to ``state_dict()`` on the model, while
        ``set_state`` should accordingly include ``load_state_dict()``.
        """
        pass

    @abstractmethod
    def set_state(self, policy_state: object) -> None:
        """Set the policy state to ``policy_state``.

        The implementation must be in correspondence with that of ``get_state``. For example, if a torch model
        is contained in the policy, ``set_state`` may include a call to ``load_state_dict()`` on the model, while
        ``get_state`` should accordingly include ``state_dict()``.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the policy state from disk."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the policy state to disk."""
        pass
