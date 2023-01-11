# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, cast

import numpy as np
import torch

from maro.rl.model import VNet
from maro.rl.policy import ContinuousRLPolicy, DiscretePolicyGradient, RLPolicy
from maro.rl.training import AbsTrainOps, BaseTrainerParams, FIFOReplayMemory, RemoteOps, SingleAgentTrainer, remote
from maro.rl.utils import TransitionBatch, discount_cumsum, get_torch_device, ndarray_to_tensor


@dataclass
class ACBasedParams(BaseTrainerParams, metaclass=ABCMeta):
    """
    Parameter bundle for Actor-Critic based algorithms (Actor-Critic & PPO)

    get_v_critic_net_func (Callable[[], VNet]): Function to get V critic net.
    grad_iters (int, default=1): Number of iterations to calculate gradients.
    critic_loss_cls (Callable, default=None): Critic loss function. If it is None, use MSE.
    lam (float, default=0.9): Lambda value for generalized advantage estimation (TD-Lambda).
    min_logp (float, default=float("-inf")): Lower bound for clamping logP values during learning.
        This is to prevent logP from becoming very large in magnitude and causing stability issues.
    """

    get_v_critic_net_func: Callable[[], VNet]
    grad_iters: int = 1
    critic_loss_cls: Optional[Callable] = None
    lam: float = 0.9
    min_logp: float = float("-inf")
    clip_ratio: Optional[float] = None


class ACBasedOps(AbsTrainOps):
    """Base class of Actor-Critic algorithm implementation. Reference: https://tinyurl.com/2ezte4cr"""

    def __init__(
        self,
        name: str,
        policy: RLPolicy,
        params: ACBasedParams,
        reward_discount: float = 0.9,
        parallelism: int = 1,
    ) -> None:
        super(ACBasedOps, self).__init__(
            name=name,
            policy=policy,
            parallelism=parallelism,
        )

        assert isinstance(self._policy, (ContinuousRLPolicy, DiscretePolicyGradient))

        self._reward_discount = reward_discount
        self._critic_loss_func = params.critic_loss_cls() if params.critic_loss_cls is not None else torch.nn.MSELoss()
        self._clip_ratio = params.clip_ratio
        self._lam = params.lam
        self._min_logp = params.min_logp
        self._v_critic_net = params.get_v_critic_net_func()
        self._is_discrete_action = isinstance(self._policy, DiscretePolicyGradient)

    def _get_critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """Compute the critic loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The critic loss of the batch.
        """
        states = ndarray_to_tensor(batch.states, device=self._device)
        returns = ndarray_to_tensor(batch.returns, device=self._device)

        self._v_critic_net.train()
        state_values = self._v_critic_net.v_values(states)
        return self._critic_loss_func(state_values, returns)

    @remote
    def get_critic_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        """Compute the critic network's gradients of a batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            grad (torch.Tensor): The critic gradient of the batch.
        """
        return self._v_critic_net.get_gradients(self._get_critic_loss(batch))

    def update_critic(self, batch: TransitionBatch) -> None:
        """Update the critic network using a batch.

        Args:
            batch (TransitionBatch): Batch.
        """
        self._v_critic_net.step(self._get_critic_loss(batch))

    def update_critic_with_grad(self, grad_dict: dict) -> None:
        """Update the critic network with remotely computed gradients.

        Args:
            grad_dict (dict): Gradients.
        """
        self._v_critic_net.train()
        self._v_critic_net.apply_gradients(grad_dict)

    def _get_actor_loss(self, batch: TransitionBatch) -> Tuple[torch.Tensor, bool]:
        """Compute the actor loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The actor loss of the batch.
            early_stop (bool): Early stop indicator.
        """
        assert isinstance(self._policy, DiscretePolicyGradient) or isinstance(self._policy, ContinuousRLPolicy)
        self._policy.train()

        states = ndarray_to_tensor(batch.states, device=self._device)
        actions = ndarray_to_tensor(batch.actions, device=self._device)
        advantages = ndarray_to_tensor(batch.advantages, device=self._device)
        logps_old = ndarray_to_tensor(batch.old_logps, device=self._device)
        if self._is_discrete_action:
            actions = actions.long()

        logps = self._policy.get_states_actions_logps(states, actions)
        if self._clip_ratio is not None:
            ratio = torch.exp(logps - logps_old)
            kl = (logps_old - logps).mean().item()
            early_stop = kl >= 0.01 * 1.5  # TODO
            clipped_ratio = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
            actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
        else:
            actor_loss = -(logps * advantages).mean()  # I * delta * log pi(a|s)
            early_stop = False

        return actor_loss, early_stop

    @remote
    def get_actor_grad(self, batch: TransitionBatch) -> Tuple[Dict[str, torch.Tensor], bool]:
        """Compute the actor network's gradients of a batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            grad (torch.Tensor): The actor gradient of the batch.
            early_stop (bool): Early stop indicator.
        """
        loss, early_stop = self._get_actor_loss(batch)
        return self._policy.get_gradients(loss), early_stop

    def update_actor(self, batch: TransitionBatch) -> bool:
        """Update the actor network using a batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            early_stop (bool): Early stop indicator.
        """
        loss, early_stop = self._get_actor_loss(batch)
        self._policy.train_step(loss)
        return early_stop

    def update_actor_with_grad(self, grad_dict_and_early_stop: Tuple[dict, bool]) -> bool:
        """Update the actor network with remotely computed gradients.

        Args:
            grad_dict_and_early_stop (Tuple[dict, bool]): Gradients and early stop indicator.

        Returns:
            early stop indicator
        """
        self._policy.train()
        self._policy.apply_gradients(grad_dict_and_early_stop[0])
        return grad_dict_and_early_stop[1]

    def get_non_policy_state(self) -> dict:
        return {
            "critic": self._v_critic_net.get_state(),
        }

    def set_non_policy_state(self, state: dict) -> None:
        self._v_critic_net.set_state(state["critic"])

    def preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        """Preprocess the batch to get the returns & advantages.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            The updated batch.
        """
        assert isinstance(batch, TransitionBatch)

        # Preprocess advantages
        states = ndarray_to_tensor(batch.states, device=self._device)  # s
        actions = ndarray_to_tensor(batch.actions, device=self._device)  # a
        if self._is_discrete_action:
            actions = actions.long()

        with torch.no_grad():
            self._v_critic_net.eval()
            self._policy.eval()
            values = self._v_critic_net.v_values(states).detach().cpu().numpy()
            values = np.concatenate([values, np.zeros(1)])
            rewards = np.concatenate([batch.rewards, np.zeros(1)])
            deltas = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]  # r + gamma * v(s') - v(s)
            batch.returns = discount_cumsum(rewards, self._reward_discount)[:-1]
            batch.advantages = discount_cumsum(deltas, self._reward_discount * self._lam)

            if self._clip_ratio is not None:
                batch.old_logps = self._policy.get_states_actions_logps(states, actions).detach().cpu().numpy()

        return batch

    def debug_get_v_values(self, batch: TransitionBatch) -> np.ndarray:
        states = ndarray_to_tensor(batch.states, device=self._device)  # s
        with torch.no_grad():
            values = self._v_critic_net.v_values(states).detach().cpu().numpy()
        return values

    def to_device(self, device: str = None) -> None:
        self._device = get_torch_device(device)
        self._policy.to_device(self._device)
        self._v_critic_net.to(self._device)


class ACBasedTrainer(SingleAgentTrainer):
    """Base class of Actor-Critic algorithm implementation.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch
        https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
    """

    def __init__(
        self,
        name: str,
        params: ACBasedParams,
        replay_memory_capacity: int = 10000,
        batch_size: int = 128,
        data_parallelism: int = 1,
        reward_discount: float = 0.9,
    ) -> None:
        super(ACBasedTrainer, self).__init__(
            name,
            replay_memory_capacity,
            batch_size,
            data_parallelism,
            reward_discount,
        )
        self._params = params

    def _register_policy(self, policy: RLPolicy) -> None:
        assert isinstance(policy, (ContinuousRLPolicy, DiscretePolicyGradient))
        self._policy = policy

    def build(self) -> None:
        self._ops = cast(ACBasedOps, self.get_ops())
        self._replay_memory = FIFOReplayMemory(
            capacity=self._replay_memory_capacity,
            state_dim=self._ops.policy_state_dim,
            action_dim=self._ops.policy_action_dim,
        )

    def _preprocess_batch(self, transition_batch: TransitionBatch) -> TransitionBatch:
        return self._ops.preprocess_batch(transition_batch)

    def get_local_ops(self) -> AbsTrainOps:
        return ACBasedOps(
            name=self._policy.name,
            policy=self._policy,
            parallelism=self._data_parallelism,
            reward_discount=self._reward_discount,
            params=self._params,
        )

    def _get_batch(self) -> TransitionBatch:
        batch = self._replay_memory.sample(-1)
        batch.advantages = (batch.advantages - batch.advantages.mean()) / batch.advantages.std()
        return batch

    def train_step(self) -> None:
        assert isinstance(self._ops, ACBasedOps)

        batch = self._get_batch()
        for _ in range(self._params.grad_iters):
            self._ops.update_critic(batch)

        for _ in range(self._params.grad_iters):
            early_stop = self._ops.update_actor(batch)
            if early_stop:
                break

    async def train_step_as_task(self) -> None:
        assert isinstance(self._ops, RemoteOps)

        batch = self._get_batch()
        for _ in range(self._params.grad_iters):
            self._ops.update_critic_with_grad(await self._ops.get_critic_grad(batch))

        for _ in range(self._params.grad_iters):
            if self._ops.update_actor_with_grad(await self._ops.get_actor_grad(batch)):  # early stop
                break
