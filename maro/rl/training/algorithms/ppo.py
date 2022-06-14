# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from maro.rl.model import VNet
from maro.rl.policy import DiscretePolicyGradient, RLPolicy
from maro.rl.training.algorithms.base import ACBasedOps, ACBasedParams, ACBasedTrainer
from maro.rl.utils import TransitionBatch, discount_cumsum, ndarray_to_tensor


@dataclass
class PPOParams(ACBasedParams):
    """Mostly inherited from `ACBasedParams`. Please refer to the doc string of `ACBasedParams`
    for more detailed information.

    clip_ratio (float, default=None): Clip ratio in the PPO algorithm (https://arxiv.org/pdf/1707.06347.pdf).
        If it is None, the actor loss is calculated using the usual policy gradient theorem.
    """

    clip_ratio: float = None

    def extract_ops_params(self) -> Dict[str, object]:
        return {
            "get_v_critic_net_func": self.get_v_critic_net_func,
            "reward_discount": self.reward_discount,
            "critic_loss_cls": self.critic_loss_cls,
            "clip_ratio": self.clip_ratio,
            "lam": self.lam,
            "min_logp": self.min_logp,
            "is_discrete_action": self.is_discrete_action,
        }

    def __post_init__(self) -> None:
        assert self.get_v_critic_net_func is not None
        assert self.clip_ratio is not None


class DiscretePPOWithEntropyOps(ACBasedOps):
    def __init__(
        self,
        name: str,
        policy_creator: Callable[[], RLPolicy],
        get_v_critic_net_func: Callable[[], VNet],
        parallelism: int = 1,
        reward_discount: float = 0.9,
        critic_loss_cls: Callable = None,
        clip_ratio: float = None,
        lam: float = 0.9,
        min_logp: float = None,
        is_discrete_action: bool = True,
    ) -> None:
        super(DiscretePPOWithEntropyOps, self).__init__(
            name=name,
            policy_creator=policy_creator,
            get_v_critic_net_func=get_v_critic_net_func,
            parallelism=parallelism,
            reward_discount=reward_discount,
            critic_loss_cls=critic_loss_cls,
            clip_ratio=clip_ratio,
            lam=lam,
            min_logp=min_logp,
            is_discrete_action=is_discrete_action,
        )
        assert is_discrete_action
        assert isinstance(self._policy, DiscretePolicyGradient)
        self._policy_old = self._policy_creator()
        self.update_policy_old()

    def update_policy_old(self) -> None:
        self._policy_old.set_state(self._policy.get_state())

    def _get_critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """Compute the critic loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The critic loss of the batch.
        """
        self._v_critic_net.train()
        states = ndarray_to_tensor(batch.states, self._device)
        state_values = self._v_critic_net.v_values(states)

        values = state_values.cpu().detach().numpy()
        values = np.concatenate([values[1:], values[-1:]])
        returns = batch.rewards + np.where(batch.terminals, 0.0, 1.0) * self._reward_discount * values
        # special care for tail state
        returns[-1] = state_values[-1]
        returns = ndarray_to_tensor(returns, self._device)

        return self._critic_loss_func(state_values.float(), returns.float())

    def _get_actor_loss(self, batch: TransitionBatch) -> Tuple[torch.Tensor, bool]:
        """Compute the actor loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The actor loss of the batch.
            early_stop (bool): Early stop indicator.
        """
        assert isinstance(self._policy, DiscretePolicyGradient)
        self._policy.train()

        states = ndarray_to_tensor(batch.states, device=self._device)
        actions = ndarray_to_tensor(batch.actions, device=self._device)
        advantages = ndarray_to_tensor(batch.advantages, device=self._device)
        logps_old = ndarray_to_tensor(batch.old_logps, device=self._device)
        if self._is_discrete_action:
            actions = actions.long()

        action_probs = self._policy.get_action_probs(states)
        dist_entropy = Categorical(action_probs).entropy()
        logps = torch.log(action_probs.gather(1, actions).squeeze())
        logps = torch.clamp(logps, min=self._min_logp, max=0.0)
        if self._clip_ratio is not None:
            ratio = torch.exp(logps - logps_old)
            clipped_ratio = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
            actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).float()
            kl = (logps_old - logps).mean().item()
            early_stop = kl >= 0.01 * 1.5  # TODO
        else:
            actor_loss = -(logps * advantages).float()  # I * delta * log pi(a|s)
            early_stop = False
        actor_loss = (actor_loss - 0.2 * dist_entropy).mean()

        return actor_loss, early_stop

    def preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        """Preprocess the batch to get the returns & advantages.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            The updated batch.
        """
        assert isinstance(batch, TransitionBatch)
        # Preprocess returns
        batch.returns = discount_cumsum(batch.rewards, self._reward_discount)

        # Preprocess advantages
        states = ndarray_to_tensor(batch.states, self._device)
        state_values = self._v_critic_net.v_values(states).cpu().detach().numpy()
        values = np.concatenate([state_values[1:], np.zeros(1).astype(np.float32)])
        deltas = batch.rewards + self._reward_discount * values - state_values
        # special care for tail state
        deltas[-1] = 0.0
        batch.advantages = discount_cumsum(deltas, self._reward_discount * self._lam)

        if self._clip_ratio is not None:
            self._policy_old.eval()
            actions = ndarray_to_tensor(batch.actions, device=self._device).long()
            batch.old_logps = self._policy_old.get_states_actions_logps(states, actions).detach().cpu().numpy()
            self._policy_old.train()

        return batch


class PPOTrainer(ACBasedTrainer):
    """PPO algorithm.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo.
    """

    def __init__(self, name: str, params: PPOParams) -> None:
        super(PPOTrainer, self).__init__(name, params)


class DiscretePPOWithEntropyTrainer(ACBasedTrainer):
    def __init__(self, name: str, params: PPOParams) -> None:
        super(DiscretePPOWithEntropyTrainer, self).__init__(name, params)

    def get_local_ops(self) -> DiscretePPOWithEntropyOps:
        return DiscretePPOWithEntropyOps(
            name=self._policy_name,
            policy_creator=self._policy_creator,
            parallelism=self._params.data_parallelism,
            **self._params.extract_ops_params(),
        )

    def train_step(self) -> None:
        assert isinstance(self._ops, DiscretePPOWithEntropyOps)
        batch = self._get_batch()
        for _ in range(self._params.grad_iters):
            self._ops.update_critic(batch)
            self._ops.update_actor(batch)
        self._ops.update_policy_old()
