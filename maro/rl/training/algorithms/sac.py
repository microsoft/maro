# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, cast

import torch

from maro.rl.model import QNet
from maro.rl.policy import ContinuousRLPolicy, RLPolicy
from maro.rl.training import AbsTrainOps, BaseTrainerParams, RandomReplayMemory, RemoteOps, SingleAgentTrainer, remote
from maro.rl.utils import TransitionBatch, get_torch_device, ndarray_to_tensor
from maro.utils import clone


@dataclass
class SoftActorCriticParams(BaseTrainerParams):
    get_q_critic_net_func: Callable[[], QNet]
    update_target_every: int = 5
    random_overwrite: bool = False
    entropy_coef: float = 0.1
    num_epochs: int = 1
    n_start_train: int = 0
    q_value_loss_cls: Optional[Callable] = None
    soft_update_coef: float = 1.0


class SoftActorCriticOps(AbsTrainOps):
    def __init__(
        self,
        name: str,
        policy: RLPolicy,
        params: SoftActorCriticParams,
        reward_discount: float = 0.9,
        parallelism: int = 1,
    ) -> None:
        super(SoftActorCriticOps, self).__init__(
            name=name,
            policy=policy,
            parallelism=parallelism,
        )

        assert isinstance(self._policy, ContinuousRLPolicy)

        self._q_net1 = params.get_q_critic_net_func()
        self._q_net2 = params.get_q_critic_net_func()
        self._target_q_net1: QNet = clone(self._q_net1)
        self._target_q_net1.eval()
        self._target_q_net2: QNet = clone(self._q_net2)
        self._target_q_net2.eval()

        self._entropy_coef = params.entropy_coef
        self._soft_update_coef = params.soft_update_coef
        self._reward_discount = reward_discount
        self._q_value_loss_func = (
            params.q_value_loss_cls() if params.q_value_loss_cls is not None else torch.nn.MSELoss()
        )

    def _get_critic_loss(self, batch: TransitionBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        self._q_net1.train()
        states = ndarray_to_tensor(batch.states, device=self._device)  # s
        next_states = ndarray_to_tensor(batch.next_states, device=self._device)  # s'
        actions = ndarray_to_tensor(batch.actions, device=self._device)  # a
        rewards = ndarray_to_tensor(batch.rewards, device=self._device)  # r
        terminals = ndarray_to_tensor(batch.terminals, device=self._device)  # d

        assert isinstance(self._policy, ContinuousRLPolicy)

        with torch.no_grad():
            next_actions, next_logps = self._policy.get_actions_with_logps(states)
            q1 = self._target_q_net1.q_values(next_states, next_actions)
            q2 = self._target_q_net2.q_values(next_states, next_actions)
            q = torch.min(q1, q2)
            y = rewards + self._reward_discount * (1.0 - terminals.float()) * (q - self._entropy_coef * next_logps)

        q1 = self._q_net1.q_values(states, actions)
        q2 = self._q_net2.q_values(states, actions)
        loss_q1 = self._q_value_loss_func(q1, y)
        loss_q2 = self._q_value_loss_func(q2, y)
        return loss_q1, loss_q2

    @remote
    def get_critic_grad(self, batch: TransitionBatch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        loss_q1, loss_q2 = self._get_critic_loss(batch)
        grad_q1 = self._q_net1.get_gradients(loss_q1)
        grad_q2 = self._q_net2.get_gradients(loss_q2)
        return grad_q1, grad_q2

    def update_critic_with_grad(self, grad_dicts: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]) -> None:
        self._q_net1.train()
        self._q_net2.train()
        self._q_net1.apply_gradients(grad_dicts[0])
        self._q_net2.apply_gradients(grad_dicts[1])

    def update_critic(self, batch: TransitionBatch) -> None:
        self._q_net1.train()
        self._q_net2.train()
        loss_q1, loss_q2 = self._get_critic_loss(batch)
        self._q_net1.step(loss_q1)
        self._q_net2.step(loss_q2)

    def _get_actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        self._policy.train()
        states = ndarray_to_tensor(batch.states, device=self._device)  # s
        actions, logps = self._policy.get_actions_with_logps(states)
        q1 = self._q_net1.q_values(states, actions)
        q2 = self._q_net2.q_values(states, actions)
        q = torch.min(q1, q2)

        loss = (self._entropy_coef * logps - q).mean()
        return loss

    @remote
    def get_actor_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        return self._policy.get_gradients(self._get_actor_loss(batch))

    def update_actor_with_grad(self, grad_dict: dict) -> None:
        self._policy.train()
        self._policy.apply_gradients(grad_dict)

    def update_actor(self, batch: TransitionBatch) -> None:
        self._policy.train()
        self._policy.train_step(self._get_actor_loss(batch))

    def get_non_policy_state(self) -> dict:
        return {
            "q_net1": self._q_net1.get_state(),
            "q_net2": self._q_net2.get_state(),
            "target_q_net1": self._target_q_net1.get_state(),
            "target_q_net2": self._target_q_net2.get_state(),
        }

    def set_non_policy_state(self, state: dict) -> None:
        self._q_net1.set_state(state["q_net1"])
        self._q_net2.set_state(state["q_net2"])
        self._target_q_net1.set_state(state["target_q_net1"])
        self._target_q_net2.set_state(state["target_q_net2"])

    def soft_update_target(self) -> None:
        self._target_q_net1.soft_update(self._q_net1, self._soft_update_coef)
        self._target_q_net2.soft_update(self._q_net2, self._soft_update_coef)

    def to_device(self, device: str = None) -> None:
        self._device = get_torch_device(device=device)
        self._q_net1.to(self._device)
        self._q_net2.to(self._device)
        self._target_q_net1.to(self._device)
        self._target_q_net2.to(self._device)


class SoftActorCriticTrainer(SingleAgentTrainer):
    def __init__(
        self,
        name: str,
        params: SoftActorCriticParams,
        replay_memory_capacity: int = 10000,
        batch_size: int = 128,
        data_parallelism: int = 1,
        reward_discount: float = 0.9,
    ) -> None:
        super(SoftActorCriticTrainer, self).__init__(
            name,
            replay_memory_capacity,
            batch_size,
            data_parallelism,
            reward_discount,
        )
        self._params = params
        self._qnet_version = self._target_qnet_version = 0

    def build(self) -> None:
        self._ops = cast(SoftActorCriticOps, self.get_ops())
        self._replay_memory = RandomReplayMemory(
            capacity=self._replay_memory_capacity,
            state_dim=self._ops.policy_state_dim,
            action_dim=self._ops.policy_action_dim,
            random_overwrite=self._params.random_overwrite,
        )

    def _register_policy(self, policy: RLPolicy) -> None:
        assert isinstance(policy, ContinuousRLPolicy)
        self._policy = policy

    def train_step(self) -> None:
        assert isinstance(self._ops, SoftActorCriticOps)

        if self._replay_memory.n_sample < self._params.n_start_train:
            print(
                f"Skip this training step due to lack of experiences "
                f"(current = {self._replay_memory.n_sample}, minimum = {self._params.n_start_train})",
            )
            return

        for _ in range(self._params.num_epochs):
            batch = self._get_batch()
            self._ops.update_critic(batch)
            self._ops.update_actor(batch)

            self._try_soft_update_target()

    async def train_step_as_task(self) -> None:
        assert isinstance(self._ops, RemoteOps)

        if self._replay_memory.n_sample < self._params.n_start_train:
            print(
                f"Skip this training step due to lack of experiences "
                f"(current = {self._replay_memory.n_sample}, minimum = {self._params.n_start_train})",
            )
            return

        for _ in range(self._params.num_epochs):
            batch = self._get_batch()
            self._ops.update_critic_with_grad(await self._ops.get_critic_grad(batch))
            self._ops.update_actor_with_grad(await self._ops.get_actor_grad(batch))

            self._try_soft_update_target()

    def _preprocess_batch(self, transition_batch: TransitionBatch) -> TransitionBatch:
        return transition_batch

    def get_local_ops(self) -> SoftActorCriticOps:
        return SoftActorCriticOps(
            name=self._policy.name,
            policy=self._policy,
            parallelism=self._data_parallelism,
            reward_discount=self._reward_discount,
            params=self._params,
        )

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._batch_size)

    def _try_soft_update_target(self) -> None:
        """Soft update the target policy and target critic."""
        self._qnet_version += 1
        if self._qnet_version - self._target_qnet_version == self._params.update_target_every:
            self._ops.soft_update_target()
            self._target_qnet_version = self._qnet_version
