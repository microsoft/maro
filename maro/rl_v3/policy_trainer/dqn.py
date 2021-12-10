from typing import Dict, Optional

import torch

from maro.rl_v3.policy import RLPolicy, ValueBasedPolicy
from maro.rl_v3.replay_memory import RandomReplayMemory
from maro.rl_v3.utils import TransitionBatch, ndarray_to_tensor
from maro.utils import clone

from .abs_trainer import SingleTrainer
from .train_worker import SingleTrainWorker


class DQNWorker(SingleTrainWorker):
    def __init__(
        self,
        name: str,
        device: torch.device,
        reward_discount: float = 0.9,
        soft_update_coef: float = 0.1,
        double: bool = False,
        enable_data_parallelism: bool = False
    ) -> None:
        super(DQNWorker, self).__init__(name, device, enable_data_parallelism)

        self._reward_discount = reward_discount
        self._soft_update_coef = soft_update_coef
        self._double = double

        self._policy: Optional[ValueBasedPolicy] = None
        self._target_policy: Optional[ValueBasedPolicy] = None

        self._loss_func = torch.nn.MSELoss()

    def _register_policy_impl(self, policy: RLPolicy) -> None:
        assert isinstance(policy, ValueBasedPolicy)

        self._policy = policy
        self._target_policy: ValueBasedPolicy = clone(policy)
        self._target_policy.set_name(f"target_{policy.name}")
        self._target_policy.eval()
        self._target_policy.to_device(self._device)

    def get_batch_grad(
        self,
        batch: TransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        assert scope == "all", f"Unrecognized scope {scope}. Excepting 'all'."

        self._policy.train()
        states = ndarray_to_tensor(batch.states, self._device)
        next_states = ndarray_to_tensor(batch.next_states, self._device)
        actions = ndarray_to_tensor(batch.actions, self._device)
        rewards = ndarray_to_tensor(batch.rewards, self._device)
        terminals = ndarray_to_tensor(batch.terminals, self._device).float()

        with torch.no_grad():
            if self._double:
                self._policy.exploit()
                actions_by_eval_policy = self._policy.get_actions_tensor(next_states)
                next_q_values = self._target_policy.q_values_tensor(next_states, actions_by_eval_policy)
            else:
                self._target_policy.exploit()
                actions = self._target_policy.get_actions_tensor(next_states)
                next_q_values = self._target_policy.q_values_tensor(next_states, actions)
        target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()

        q_values = self._policy.q_values_tensor(states, actions)
        loss: torch.Tensor = self._loss_func(q_values, target_q_values)

        return {"grad": self._policy.get_gradients(loss)}

    def get_worker_state_dict(self, scope: str = "all") -> dict:
        return {
            "policy_state": self._policy.get_policy_state(),
            "target_policy_state": self._target_policy.get_policy_state()
        }

    def set_worker_state_dict(self, worker_state_dict: dict, scope: str = "all") -> None:
        self._policy.set_policy_state(worker_state_dict["policy_state"])
        self._target_policy.set_policy_state(worker_state_dict["target_policy_state"])

    def update(self) -> None:
        grad_dict = self._get_batch_grad(self._batch)

        self._policy.train()
        self._policy.apply_gradients(grad_dict["grad"])

    def soft_update_target(self) -> None:
        self._target_policy.soft_update(self._policy, self._soft_update_coef)


class DQN(SingleTrainer):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        name (str): Unique identifier for the policy.
        policy (ValueBasedPolicy): The policy to be trained.
        replay_memory_capacity (int): Capacity of the replay memory. Defaults to 100000.
        train_batch_size (int): Batch size for training the Q-net. Defaults to 128.
        num_epochs (int): Number of training epochs per call to ``learn``. Defaults to 1.
        reward_discount (float): Reward decay as defined in standard RL terminology. Defaults to 0.9.
        update_target_every (int): Number of gradient steps between target model updates. Defaults to 5.
        soft_update_coef (float): Soft update coefficient, e.g.,
            target_model = (soft_update_coef) * eval_model + (1-soft_update_coef) * target_model.
            Defaults to 0.1.
        double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
        random_overwrite (bool): This specifies overwrite behavior when the replay memory capacity is reached. If True,
            overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
            wrap-around. Defaults to False.
        device (str): Identifier for the torch device. The policy will be moved to the specified device. If it is
            None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise. Defaults to None.
        enable_data_parallelism (bool): Whether to enable data parallelism in this trainer. Defaults to False.
    """
    def __init__(
        self,
        name: str,
        policy: ValueBasedPolicy = None,
        replay_memory_capacity: int = 100000,
        train_batch_size: int = 128,
        num_epochs: int = 1,
        reward_discount: float = 0.9,
        update_target_every: int = 5,
        soft_update_coef: float = 0.1,
        double: bool = False,
        random_overwrite: bool = False,
        device: str = None,
        enable_data_parallelism: bool = False
    ) -> None:
        super(DQN, self).__init__(
            name=name,
            device=device,
            enable_data_parallelism=enable_data_parallelism,
            train_batch_size=train_batch_size
        )

        self._replay_memory_capacity = replay_memory_capacity
        self._random_overwrite = random_overwrite
        if policy is not None:
            self.register_policy(policy)

        self._train_batch_size = train_batch_size
        self._num_epochs = num_epochs
        self._reward_discount = reward_discount

        self._update_target_every = update_target_every
        self._soft_update_coef = soft_update_coef
        self._double = double

        self._loss_func = torch.nn.MSELoss()
        self._policy_version = self._target_policy_version = 0

    def _train_step_impl(self) -> None:
        for _ in range(self._num_epochs):
            self._worker.set_batch(self._get_batch())
            self._worker.update()
        self._try_soft_update_target()

    def _register_policy_impl(self, policy: ValueBasedPolicy) -> None:
        self._worker = DQNWorker(
            name="worker", device=self._device, reward_discount=self._reward_discount,
            soft_update_coef=self._soft_update_coef, double=self._double,
            enable_data_parallelism=self._enable_data_parallelism
        )
        self._worker.register_policy(policy)

        self._replay_memory = RandomReplayMemory(
            capacity=self._replay_memory_capacity, state_dim=policy.state_dim,
            action_dim=policy.action_dim, random_overwrite=self._random_overwrite
        )

    def _try_soft_update_target(self) -> None:
        self._policy_version += 1
        if self._policy_version - self._target_policy_version == self._update_target_every:
            self._worker.soft_update_target()
            self._target_policy_version = self._policy_version

    def get_policy_state(self) -> object:
        return self._worker.get_policy_state()

    def set_policy_state(self, policy_state: object) -> None:
        self._worker.set_policy_state(policy_state)
