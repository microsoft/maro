# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import scipy
import torch

from maro.rl.model import VNet
from maro.rl.policy import ContinuousRLPolicy, DiscretePolicyGradient, RLPolicy
from maro.rl.training import AbsTrainer, AbsTrainOps, TrainerParams, remote
from maro.rl.utils import TransitionBatch, discount_cumsum, get_torch_device, ndarray_to_tensor


@dataclass
class TRPOParams(TrainerParams):

    # maro common params
    get_v_critic_net_func: Callable[[], VNet] = None
    grad_iters: int = 1
    critic_loss_cls: Optional[Callable] = None

    num_epochs: int = 1
    update_target_every: int = 5
    random_overwrite: bool = False

    # Refer to Spinning Up params,
    # steps_per_epoch: int = 4000
    # epochs: int = 50
    # vf_lr: float = 1e-3
    # train_v_iters: int = 80
    # cg_iters: int = 10
    # backtrack_iters: int = 10
    # max_ep_len: int = 1000
    # logger_kwargs: dict = dict()
    # save_freq: int = 10

    ac_kwargs: dict = dict()
    gamma: float = 0.99
    gae_lambda: float = 0.95
    damping_coeff: float = 0.1
    max_kl: float = 0.01
    backtrack_coeff: float = 0.8
    optim_critic_iters: int = 5
    max_backtracks: int = 10


class TRPOOps(AbsTrainOps):
    def __init__(
        self,
        name: str,
        policy: RLPolicy,
        params: TRPOParams,
    ) -> None:
        super(TRPOOps, self).__init__(
            name=name,
            policy=policy,
        )

        # TRPO can be used for environments with either discrete or continuous action spaces.
        assert isinstance(self._policy, (ContinuousRLPolicy, DiscretePolicyGradient))

        self._v_critic_net = params.get_v_critic_net_func()
        self._critic_loss_func = params.critic_loss_cls() if params.critic_loss_cls is not None else torch.nn.MSELoss()
        # self._is_discrete_action = isinstance(self._policy, DiscretePolicyGradient)
        self._reward_discount = params.gamma
        self._damping = params.damping_coeff
        self._delta = params.max_kl
        self._lambda = params.gae_lambda
        self._backtrack_coeff = params.backtrack_coeff
        self._optim_critic_iters = params.optim_critic_iters
        self._max_backtracks = params.max_backtracks
        self._optim_critic_iters = params.optim_critic_iters

    def _get_critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """Compute the critic loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The critic loss of the batch.
        """
        # TODO:refer to maro-ppo

        self._v_critic_net.train()
        states = ndarray_to_tensor(batch.states, self._device)
        # value
        state_values = self._v_critic_net.v_values(states)

        values = state_values.cpu().detach().numpy()
        values = np.concatenate([values[1:], values[-1:]])
        returns = batch.rewards + np.where(batch.terminals, 0.0, 1.0) * self._reward_discount * values
        returns[-1] = state_values[-1]
        returns = ndarray_to_tensor(returns, self._device)

        # value_loss = (state_values - returns).pow(2).mean()

        return self._critic_loss_func(state_values.float(), returns.float())

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
        # TODO
        # self.optim.zero_grad()
        # vf_loss.backward()
        # self.optim.step()
        # where to run up 3 oprations?

        self._v_critic_net.step(self._get_critic_loss(batch))

    def update_critic_with_grad(self, grad_dict: dict) -> None:
        """Update the critic network with remotely computed gradients.

        Args:
            grad_dict (dict): Gradients.
        """
        self._v_critic_net.train()
        self._v_critic_net.apply_gradients(grad_dict)

    def _get_actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """Compute the actor loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The actor loss of the batch.
        """
        assert isinstance(self._policy, (ContinuousRLPolicy, DiscretePolicyGradient))
        self._policy.train()

        # calculate villia gradient
        # states actions advantages logp_old
        states = ndarray_to_tensor(batch.states, device=self._device)
        actions = ndarray_to_tensor(batch.actions, device=self._device)
        advantages = ndarray_to_tensor(batch.advantages, device=self._device)
        logp_old = ndarray_to_tensor(batch.old_logps, device=self._device)
        if self._is_discrete_action:
            actions = actions.long()
        # logps
        action_probs = self._policy.get_action_probs(states)
        logps = torch.log(action_probs.gather(1, actions).squeeze())  # 维度调整
        logps = torch.clamp(logps, min=self._min_logp, max=0.0)  # 压缩到指定范围
        # actor loss
        ratio = torch.exp(logps - logp_old)  # pi(a|s) / pi_old(a|s)
        actor_loss = -torch.mean(ratio * advantages)  # -(ratio * advantages).mean()
        # gradients
        # compute in get_actor_grad(use the result of _get_actor_loss) func
        return actor_loss

    @remote
    def get_actor_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        # step 7 of Pseudocode in Spinning Up --> get gradient
        """Compute the actor network's gradients of a batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            grad (torch.Tensor): The actor gradient of the batch.
        """
        return self._policy.get_gradients(self._get_actor_loss(batch))

    def update_actor(self, batch: TransitionBatch) -> None:
        """Update the actor network using a batch.

        Args:
            batch (TransitionBatch): Batch.
        """
        self._policy.train()  # TODO:Does this step need to be performed?
        self._policy.train_step(self._get_actor_loss(batch))

    def update_actor_with_grad(self, grad_dict: dict) -> None:
        """Update the actor network with remotely computed gradients.

        Args:
            grad_dict (dict): Gradients.
        """
        self._policy.train()
        self._policy.apply_gradients(grad_dict)

    def _get_kl_divergence(self, batch: TransitionBatch):
        logp_old = ndarray_to_tensor(batch.old_logps, device=self._device)
        states = ndarray_to_tensor(batch.states, device=self._device)
        actions = ndarray_to_tensor(batch.actions, device=self._device)
        action_probs = self._policy.get_action_probs(states)
        logps = torch.log(action_probs.gather(1, actions).squeeze())  # 维度调整
        logps = torch.clamp(logps, min=self._min_logp, max=0.0)  # 压缩到指定范围
        kl = scipy.stats.entropy(logps, logp_old)  # 概率分布求log得dist，暂用scipy函数直接求kl
        return kl

    def get_kl_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        return self._policy.get_gradients(self._get_kl_divergence(batch).mean())

    def _MVP(self, v: torch.Tensor, batch: TransitionBatch) -> torch.Tensor:
        """Matrix vector product."""
        # caculate second order gradient of kl with respect to theta
        kl_v = (self.get_kl_grad(batch) * v).sum()
        flat_kl_grad_grad = self._policy.get_gradients(kl_v).detach()
        return flat_kl_grad_grad + v * self._damping

    def _conjugate_gradients(self, batch: TransitionBatch, nsteps: int = 10, residual_tol: float = 1e-10):
        x = torch.zeros_like(batch)
        r = batch.clone()
        p = batch.clone()
        rdotr = torch.dot(r, r)
        for _ in range(nsteps):
            z = self._MVP(p, self.get_kl_grad(batch))
            alpha = rdotr / p.dot(z)
            x += alpha * p
            r -= alpha * z
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            p = r + new_rdotr / rdotr * p
            rdotr = new_rdotr
        return x

    def get_search_direction(self, batch: TransitionBatch, nsteps=10):
        # step 8 of Pseudocode in Spinning Up --> get x(search_direction)
        search_direction = -self._conjugate_gradients(
            self.get_actor_grad(batch),
            self.get_kl_grad(batch),
            nsteps,
        )
        return search_direction

    def get_step_size(self, batch: TransitionBatch):
        search_direction = self.get_search_direction(batch)
        step_size = torch.sqrt(
            2 * self._delta / (search_direction * self._MVP(search_direction, batch)).sum(0, keepdim=True),
        )
        return step_size

    def _set_from_flat_params(self, policy, flat_params: torch.Tensor):
        # TODO
        # prev_idx = 0
        # for param in policy.parameters():# TODO model.parameters()
        #     flat_size = int(np.prod(list(param.size())))
        #     param.data.copy_(
        #         flat_params[prev_idx:prev_idx + flat_size].view(param.size())
        #     )
        #     prev_idx += flat_size
        return policy

    def get_linesearch_stepsize(self, batch: TransitionBatch):
        # step 9 of Pseudocode in Spinning Up --> update policy by backtracking line search
        with torch.no_grad():
            flat_params = torch.cat(
                [param.data.view(-1) for param in self._policy.parameters()],
            )  # TODO self._policy.parameters()
            for i in range(self._max_backtracks):
                new_flat_params = flat_params + self.get_step_size(batch) * self.get_search_direction(batch)
                old_actor_loss = self._get_actor_loss(batch)
                self._set_from_flat_params(self._policy, new_flat_params)
                kl = self._get_kl_divergence(batch).mean()
                new_actor_loss = self._get_actor_loss(batch)
                if kl < self._delta and new_actor_loss < old_actor_loss:
                    if i > 0:
                        print()
                        # warnings.warn(f"Backtracking to step {i}.")
                    break
                elif i < self._max_backtracks - 1:
                    step_size = self.get_step_size(batch)
                    step_size = step_size * self._backtrack_coeff
                else:
                    self._set_from_flat_params(self._policy, new_flat_params)
                    step_size = torch.tensor([0.0])
                    # warnings.warn(
                    #     "Line search failed! It seems hyperparamters" " are poor and need to be changed.",
                    # )
        return step_size

    def get_non_policy_state(self) -> dict:
        return {
            "critic": self._v_critic_net.get_state(),
        }

    def set_non_policy_state(self, state: dict) -> None:
        self._v_critic_net.set_state(state["critic"])

    def to_device(self, device: str = None) -> None:
        self._device = get_torch_device(device)
        self._policy.to_device(self._device)
        self._v_critic_net.to(self._device)

    def preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        """Preprocess the batch to get the returns & advantages.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            The updated batch.
        """
        assert isinstance(batch, TransitionBatch)
        # TODO:(maybe)Done. Refer to Spinning Up tf1-TRPO GAEBuffer and maro-ppo.
        # In fact, they are same in Spinning Up Buffer

        # Preprocess advantages
        states = ndarray_to_tensor(batch.states, device=self._device)  # s
        actions = ndarray_to_tensor(batch.actions, device=self._device)  # a
        if self._is_discrete_action:
            actions = actions.long()

        with torch.no_grad():
            self._v_critic_net.eval()
            self._policy.eval()
            values = self._v_critic_net.v_values(states).detach().cpu().numpy()
            values = np.concatenate([values, np.zeros(1)])  # vals
            rewards = np.concatenate([batch.rewards, np.zeros(1)])  # rews
            deltas = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]  # delta = r + gamma * v(s') - v(s)
            batch.returns = discount_cumsum(rewards, self._reward_discount)[:-1]  # rews, gamma
            batch.advantages = discount_cumsum(deltas, self._reward_discount * self._lam)  # delta, gamma * lam

        return batch


class TRPOTrainer(AbsTrainer):
    """TRPO algorithm.

    References:
        https://
    """

    def __init__(
        self,
        name: str,
        params: TRPOParams,
        replay_memory_capacity: int = 10000,
        batch_size: int = 128,
        data_parallelism: int = 1,
        reward_discount: float = 0.9,
    ) -> None:
        super(TRPOTrainer, self).__init__(
            name,
            params,
            replay_memory_capacity,
            batch_size,
            data_parallelism,
            reward_discount,
        )
        self._params = params

    def _register_policy(self, policy: RLPolicy) -> None:
        assert isinstance(self._policy, (ContinuousRLPolicy, DiscretePolicyGradient))
        self._policy = policy

    def _preprocess_batch(self, transition_batch: TransitionBatch) -> TransitionBatch:
        self._ops.preprocess_batch(transition_batch)

    def get_local_ops(self) -> AbsTrainOps:
        return TRPOOps(
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
        assert isinstance(self._ops, TRPOOps)
        # TODO:move batch = self._get_batch() to for loop?
        batch = self._get_batch()
        for _ in range(self._params.grad_iters):
            self._ops.update_critic(batch)
            self._ops.update_actor(batch)

    # async def train_step_as_task(self) -> None:
    #     assert isinstance(self._ops, RemoteOps)
    #     pass
