from typing import Callable, Optional

import numpy as np
import torch

from maro.rl.utils import discount_cumsum
from maro.rl_v3.model import VNet
from maro.rl_v3.policy import DiscretePolicyGradient
from maro.rl_v3.replay_memory import FIFOReplayMemory
from maro.rl_v3.utils import TransitionBatch
from maro.utils import clone
from .abs_trainer import SingleTrainer


class DiscreteActorCritic(SingleTrainer):
    """
    TODO: docs.
    """
    def __init__(
        self,
        name: str,
        get_v_critic_net_func: Callable[[], VNet],
        policy: DiscretePolicyGradient = None,
        replay_memory_capacity: int = 10000,
        train_batch_size: int = 128,
        grad_iters: int = 1,
        reward_discount: float = 0.9,
        lam: float = 0.9,
        clip_ratio: float = None,
        critic_loss_cls: Callable = None,
        min_logp: float = None
    ) -> None:
        super(DiscreteActorCritic, self).__init__(name)

        self._replay_memory_capacity = replay_memory_capacity

        self._get_v_net_func = get_v_critic_net_func
        self._policy: Optional[DiscretePolicyGradient] = None
        self._v_critic_net: Optional[VNet] = None
        if policy is not None:
            self.register_policy(policy)

        self._train_batch_size = train_batch_size
        self._reward_discount = reward_discount
        self._lam = lam
        self._clip_ratio = clip_ratio
        self._min_logp = min_logp
        self._grad_iters = grad_iters

        self._critic_loss_func = critic_loss_cls() if critic_loss_cls is not None else torch.nn.MSELoss()

    def _record_impl(self, policy_name: str, transition_batch: TransitionBatch) -> None:
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def register_policy(self, policy: DiscretePolicyGradient) -> None:
        assert isinstance(policy, DiscretePolicyGradient)
        self._policy = policy
        self._replay_memory = FIFOReplayMemory(
            capacity=self._replay_memory_capacity, state_dim=policy.state_dim,
            action_dim=policy.action_dim
        )
        self._v_critic_net = self._get_v_net_func()

    def train_step(self) -> None:
        self._improve(self._get_batch())

    def _improve(self, batch: TransitionBatch) -> None:
        v_critic_net_copy = clone(self._v_critic_net)
        v_critic_net_copy.eval()

        states = self._policy.ndarray_to_tensor(batch.states)
        actions = self._policy.ndarray_to_tensor(batch.actions).long()

        self._policy.eval()
        logps_old = self._policy.get_state_action_logps(states, actions)  # [B], action log-probability when sampling

        self._policy.train()
        self._v_critic_net.train()
        for _ in range(self._grad_iters):
            values = self._v_critic_net.v_values(states).detach().numpy()
            values = np.concatenate([values, values[-1:]])
            rewards = np.concatenate([batch.rewards, values[-1:]])
            deltas = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]
            returns = self._policy.ndarray_to_tensor(discount_cumsum(rewards, self._reward_discount)[:-1])
            advantages = self._policy.ndarray_to_tensor(discount_cumsum(deltas, self._reward_discount * self._lam))

            # Critic loss
            state_values = self._v_critic_net.v_values(states)  # [B], state values given by critic
            critic_loss = self._critic_loss_func(state_values, returns)

            # Actor loss
            action_probs = self._policy.get_action_probs(states)
            logps = torch.log(action_probs.gather(1, actions).squeeze())  # [B]
            logps = torch.clamp(logps, min=self._min_logp, max=.0)
            if self._clip_ratio is not None:
                ratio = torch.exp(logps - logps_old)
                clipped_ratio = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
                actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
            else:
                actor_loss = -(logps * advantages).mean()

            # Update
            self._policy.step(actor_loss)
            self._v_critic_net.step(critic_loss * 0.1)  # TODO
    #
    # def improve_backup_version(self, batch: TransitionBatch) -> None:
    #     self._policy.train()
    #     self._v_critic_net.train()
    #
    #     v_critic_net_copy = clone(self._v_critic_net)
    #     v_critic_net_copy.eval()
    #
    #     states = self._policy.ndarray_to_tensor(batch.states)
    #     actions = self._policy.ndarray_to_tensor(batch.actions).long()
    #     logps_old = self._policy.ndarray_to_tensor(batch.logps)  # [B], action log-probability when sampling
    #     values = v_critic_net_copy.v_values(states).detach().numpy()
    #     values = np.concatenate([values, values[-1:]])
    #     rewards = np.concatenate([batch.rewards, values[-1:]])
    #     deltas = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]
    #     returns = self._policy.ndarray_to_tensor(discount_cumsum(rewards, self._reward_discount)[:-1])
    #     advantages = self._policy.ndarray_to_tensor(discount_cumsum(deltas, self._reward_discount * self._lam))
    #
    #     for _ in range(self._grad_iters):
    #         # Critic loss
    #         state_values = self._v_critic_net.v_values(states)  # [B], state values given by critic
    #         critic_loss = self._critic_loss_func(state_values, returns)
    #
    #         # Actor loss
    #         action_probs = self._policy.get_action_probs(states)
    #         logps = torch.log(action_probs.gather(1, actions).squeeze())  # [B]
    #         logps = torch.clamp(logps, min=self._min_logp, max=.0)
    #         if self._clip_ratio is not None:
    #             ratio = torch.exp(logps - logps_old)
    #             clipped_ratio = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
    #             actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
    #         else:
    #             actor_loss = -(logps * advantages).mean()
    #
    #         # Update
    #         self._policy.step(actor_loss)
    #         self._v_critic_net.step(critic_loss * 0.1)
