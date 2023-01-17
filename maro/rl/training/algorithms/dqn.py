# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Dict, cast

import torch

from maro.rl.policy import RLPolicy, ValueBasedPolicy
from maro.rl.training import AbsTrainOps, BaseTrainerParams, RandomReplayMemory, RemoteOps, SingleAgentTrainer, remote
from maro.rl.utils import TransitionBatch, get_torch_device, ndarray_to_tensor
from maro.utils import clone


@dataclass
class DQNParams(BaseTrainerParams):
    """
    num_epochs (int, default=1): Number of training epochs.
    update_target_every (int, default=5): Number of gradient steps between target model updates.
    soft_update_coef (float, default=0.1): Soft update coefficient, e.g.,
        target_model = (soft_update_coef) * eval_model + (1-soft_update_coef) * target_model.
    double (bool, default=False): If True, the next Q values will be computed according to the double DQN algorithm,
        i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
        See https://arxiv.org/pdf/1509.06461.pdf for details.
    random_overwrite (bool, default=False): This specifies overwrite behavior when the replay memory capacity
        is reached. If True, overwrite positions will be selected randomly. Otherwise, overwrites will occur
        sequentially with wrap-around.
    """

    num_epochs: int = 1
    update_target_every: int = 5
    soft_update_coef: float = 0.1
    double: bool = False
    random_overwrite: bool = False


class DQNOps(AbsTrainOps):
    def __init__(
        self,
        name: str,
        policy: RLPolicy,
        params: DQNParams,
        reward_discount: float = 0.9,
        parallelism: int = 1,
    ) -> None:
        super(DQNOps, self).__init__(
            name=name,
            policy=policy,
            parallelism=parallelism,
        )

        assert isinstance(self._policy, ValueBasedPolicy)

        self._reward_discount = reward_discount
        self._soft_update_coef = params.soft_update_coef
        self._double = params.double
        self._loss_func = torch.nn.MSELoss()

        self._target_policy: ValueBasedPolicy = clone(self._policy)
        self._target_policy.set_name(f"target_{self._policy.name}")
        self._target_policy.eval()

    def _get_batch_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """Compute the loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The loss of the batch.
        """
        assert isinstance(batch, TransitionBatch)
        assert isinstance(self._policy, ValueBasedPolicy)

        self._policy.train()
        states = ndarray_to_tensor(batch.states, device=self._device)
        next_states = ndarray_to_tensor(batch.next_states, device=self._device)
        actions = ndarray_to_tensor(batch.actions, device=self._device)
        rewards = ndarray_to_tensor(batch.rewards, device=self._device)
        terminals = ndarray_to_tensor(batch.terminals, device=self._device).float()

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
        return self._loss_func(q_values, target_q_values)

    @remote
    def get_batch_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        """Compute the network's gradients of a batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            grad (torch.Tensor): The gradient of the batch.
        """
        return self._policy.get_gradients(self._get_batch_loss(batch))

    def update_with_grad(self, grad_dict: dict) -> None:
        """Update the network with remotely computed gradients.

        Args:
            grad_dict (dict): Gradients.
        """
        self._policy.train()
        self._policy.apply_gradients(grad_dict)

    def update(self, batch: TransitionBatch) -> None:
        """Update the network using a batch.

        Args:
            batch (TransitionBatch): Batch.
        """
        self._policy.train()
        self._policy.train_step(self._get_batch_loss(batch))

    def get_non_policy_state(self) -> dict:
        return {
            "target_q_net": self._target_policy.get_state(),
        }

    def set_non_policy_state(self, state: dict) -> None:
        self._target_policy.set_state(state["target_q_net"])

    def soft_update_target(self) -> None:
        """Soft update the target policy."""
        self._target_policy.soft_update(self._policy, self._soft_update_coef)

    def to_device(self, device: str = None) -> None:
        self._device = get_torch_device(device)
        self._policy.to_device(self._device)
        self._target_policy.to_device(self._device)


class DQNTrainer(SingleAgentTrainer):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.
    """

    def __init__(
        self,
        name: str,
        params: DQNParams,
        replay_memory_capacity: int = 10000,
        batch_size: int = 128,
        data_parallelism: int = 1,
        reward_discount: float = 0.9,
    ) -> None:
        super(DQNTrainer, self).__init__(
            name,
            replay_memory_capacity,
            batch_size,
            data_parallelism,
            reward_discount,
        )
        self._params = params
        self._q_net_version = self._target_q_net_version = 0

    def build(self) -> None:
        self._ops = cast(DQNOps, self.get_ops())
        self._replay_memory = RandomReplayMemory(
            capacity=self._replay_memory_capacity,
            state_dim=self._ops.policy_state_dim,
            action_dim=self._ops.policy_action_dim,
            random_overwrite=self._params.random_overwrite,
        )

    def _register_policy(self, policy: RLPolicy) -> None:
        assert isinstance(policy, ValueBasedPolicy)
        self._policy = policy

    def _preprocess_batch(self, transition_batch: TransitionBatch) -> TransitionBatch:
        return transition_batch

    def get_local_ops(self) -> AbsTrainOps:
        return DQNOps(
            name=self._policy.name,
            policy=self._policy,
            parallelism=self._data_parallelism,
            reward_discount=self._reward_discount,
            params=self._params,
        )

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._batch_size)

    def train_step(self) -> None:
        assert isinstance(self._ops, DQNOps)
        for _ in range(self._params.num_epochs):
            self._ops.update(self._get_batch())

        self._try_soft_update_target()

    async def train_step_as_task(self) -> None:
        assert isinstance(self._ops, RemoteOps)
        for _ in range(self._params.num_epochs):
            batch = self._get_batch()
            self._ops.update_with_grad(await self._ops.get_batch_grad(batch))

        self._try_soft_update_target()

    def _try_soft_update_target(self) -> None:
        """Soft update the target policy and target critic."""
        self._q_net_version += 1
        if self._q_net_version - self._target_q_net_version == self._params.update_target_every:
            self._ops.soft_update_target()
            self._target_q_net_version = self._q_net_version
