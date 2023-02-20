# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, cast, Union, Any

import numpy as np

from maro.rl.model import VNet
from maro.rl.policy import ContinuousRLPolicy, DiscretePolicyGradient, RLPolicy
from maro.rl.training import AbsTrainOps, BaseTrainerParams, FIFOReplayMemory, RemoteOps, SingleAgentTrainer, remote
from maro.rl.utils import TransitionBatch, discount_cumsum, get_torch_device, ndarray_to_tensor
from maro.rl.training.algorithms.base.trpo_base import Policy
import warnings
from typing import Any, Dict, List, Optional, Type, Union
from torch.distributions import Independent, Normal, Distribution
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.distributions import kl_divergence


@dataclass
class TRPOParams(BaseTrainerParams, metaclass=ABCMeta):
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

    clip_ratio: Optional[float] = None
    max_kl: float = 0.01,
    backtrack_coeff: float = 0.8,
    max_backtracks: int = 10


class TRPOOps(AbsTrainOps):
    """Base class of Actor-Critic algorithm implementation. Reference: https://tinyurl.com/2ezte4cr"""

    def __init__(
        self,
        name: str,
        policy: ContinuousRLPolicy,
        params: TRPOParams,
        dist_fn: Type[torch.distributions.Distribution],
        reward_discount: float = 0.9,
        parallelism: int = 1,
        deterministic_eval: bool = False,

    ) -> None:
        super(TRPOOps, self).__init__(
            name=name,
            policy=policy,
            parallelism=parallelism,
        )

        assert isinstance(self._policy, (ContinuousRLPolicy, DiscretePolicyGradient))

        self._reward_discount = reward_discount
        self._critic_loss_func = params.critic_loss_cls() if params.critic_loss_cls is not None else torch.nn.MSELoss()
        self._clip_ratio = params.clip_ratio
        self._v_critic_net = params.get_v_critic_net_func()
        self._is_discrete_action = isinstance(self._policy, DiscretePolicyGradient)
        self._max_backtracks = params.max_backtracks
        self._deterministic_eval = deterministic_eval
        self.dist_fn = dist_fn
        self.policy_net = Policy(self.policy_state_dim, self.policy_action_dim)
        self._delta = 0.005

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



    # def _get_flat_grad(
    #     self, y: torch.Tensor, model: nn.Module, **kwargs: Any
    # ) -> torch.Tensor:
    #     grads = torch.autograd.grad(y, model.parameters(), **kwargs)  # type: ignore
    #     return torch.cat([grad.reshape(-1) for grad in grads])

    def get_kl(self, states):
        mean1, log_std1, std1 = self.policy_net(Variable(states))
        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def _MVP(self, v: torch.Tensor, flat_kl_grad: torch.Tensor) -> torch.Tensor:
        """Matrix vector product."""
        # caculate second order gradient of kl with respect to theta
        kl_v = (flat_kl_grad * v).sum()
        kl_v = torch.zeros(1, requires_grad=True)
        flat_grads = self._v_critic_net.get_gradients(kl_v)
        flat_grad_list = [y for x, y in flat_grads.items()]
        flat_kl_grad_grad = torch.cat([grad.reshape(-1) for grad in flat_grad_list]).detach()
        return flat_kl_grad_grad + v * 0.1

    def _conjugate_gradients(
        self,
        minibatch: torch.Tensor,
        flat_kl_grad: torch.Tensor,
        nsteps: int = 10,
        residual_tol: float = 1e-10
    ) -> torch.Tensor:
        x = torch.zeros_like(minibatch)
        r, p = minibatch.clone(), minibatch.clone()
        # Note: should be 'r, p = minibatch - MVP(x)', but for x=0, MVP(x)=0.
        # Change if doing warm start.
        rdotr = r.dot(r)
        for _ in range(nsteps):
            z = self._MVP(p, flat_kl_grad)
            alpha = rdotr / p.dot(z)
            x += alpha * p
            r -= alpha * z
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            p = r + new_rdotr / rdotr * p
            rdotr = new_rdotr
        return x

    def _set_from_flat_params(
        self, model: nn.Module, flat_params: torch.Tensor
    ) -> nn.Module:
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size())
            )
            prev_ind += flat_size
        return model

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

        reward = torch.Tensor(actions.size(0),1)
        # ratio:
        if self._is_discrete_action:
            actions = actions.long()
        logps = self._policy.get_states_actions_logps(states, actions)
        ratio = torch.exp(logps - logps_old)
        ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
        # actor_loss
        actor_loss = -(ratio * reward).mean()

        # flat_gard
        flat_grads = self._policy.get_gradients(actor_loss)
        flat_grad_list = [y for x, y in flat_grads.items()]
        flat_grads = torch.cat([grad.reshape(-1) for grad in flat_grad_list])

        # calculate natural gradient
        with torch.no_grad():
            self._policy.train()
            states = ndarray_to_tensor(batch.states, device=self._device)

        #  kl
        """
         因类型不同，所不采用torch中的 kl_divergence
         get_kl 源于 https://github.com/ikostrikov/pytorch-trpo/blob/master/trpo.py
        """
        kl = self.get_kl(states).mean()
        # kl = kl_divergence(old_logps, logps).mean()

        # kl flat_gard
        flat_kl_grad = self._policy.get_gradients(kl)
        flat_kl_grad_list = [y for x, y in flat_kl_grad.items()]
        flat_kl_grad = torch.cat([grad.reshape(-1) for grad in flat_kl_grad_list])

        search_direction = -self._conjugate_gradients(
            flat_grads,
            flat_kl_grad,
            nsteps=10
        )

        # stepsize: calculate max stepsize constrained by kl bound
        step_size = torch.sqrt(
            2 * self._delta /
            (search_direction *
             self._MVP(search_direction, flat_kl_grad)).sum(0, keepdim=True)
        )

        # stepsize: linesearch stepsize
        with torch.no_grad():

            flat_params = torch.cat(
                [param.data.view(-1) for param in self.policy_net.parameters()]
            )

            for i in range(self._max_backtracks):
                print(flat_params)
                print(step_size * search_direction)
                new_flat_params = flat_params + step_size * search_direction
                self._set_from_flat_params(self.policy_net, new_flat_params)
                # calculate kl and if in bound, loss actually down
                new_logps = self._policy.get_actions_with_logps(states)[1]
                new_ratio = torch.exp(new_logps - logps_old)
                new_ratio = new_ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                new_actor_loss = -(new_ratio * advantages).mean()
                kl = self.get_kl(states).mean()
                # kl = kl_divergence(old_dist, new_dist).mean()

                if kl < self._delta and new_actor_loss < actor_loss:
                    if i > 0:
                        warnings.warn(f"Backtracking to step {i}.")
                    break
                elif i < self._max_backtracks - 1:
                    step_size = step_size * 0.8
                else:
                    self._set_from_flat_params(self.policy_net, new_flat_params)
                    step_size = torch.tensor([0.0])
                    warnings.warn(
                        "Line search failed! It seems hyperparamters"
                        " are poor and need to be changed."
                    )

        return actor_loss

    @remote
    def get_actor_grad(self, batch: TransitionBatch) -> Tuple[Dict[str, torch.Tensor], bool]:
        """Compute the actor network's gradients of a batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            grad (torch.Tensor): The actor gradient of the batch.
            early_stop (bool): Early stop indicator.
        """
        loss = self._get_actor_loss(batch)
        return self._policy.get_gradients(loss)

    def update_actor(self, batch: TransitionBatch) -> bool:
        """Update the actor network using a batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            early_stop (bool): Early stop indicator.
        """
        loss = self._get_actor_loss(batch)
        self._policy.train_step(loss)

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


class TRPOTrainer(SingleAgentTrainer):
    """Base class of Actor-Critic algorithm implementation.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch
        https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
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
        self._ops = cast(TRPOOps, self.get_ops())
        self._replay_memory = FIFOReplayMemory(
            capacity=self._replay_memory_capacity,
            state_dim=self._ops.policy_state_dim,
            action_dim=self._ops.policy_action_dim,
        )

    def _preprocess_batch(self, transition_batch: TransitionBatch) -> TransitionBatch:
        return self._ops.preprocess_batch(transition_batch)

    def get_local_ops(self) -> AbsTrainOps:
        return TRPOOps(
            name=self._policy.name,
            policy=self._policy,
            parallelism=self._data_parallelism,
            reward_discount=self._reward_discount,
            params=self._params,
            dist_fn=Type[torch.distributions.Distribution],
        )

    def _get_batch(self) -> TransitionBatch:
        batch = self._replay_memory.sample(-1)
        # RuntimeError: gather_out_cpu(): Expected dtype int64 for index
        np.seterr(divide='ignore', invalid='ignore')
        batch.advantages = (batch.advantages - batch.advantages.mean()) / batch.advantages.std()
        return batch

    def train_step(self) -> None:
        assert isinstance(self._ops, TRPOOps)
        batch = self._get_batch()

        for _ in range(self._params.grad_iters):
            self._ops.update_critic(batch)
            self._ops.update_actor(batch)

    async def train_step_as_task(self) -> None:
        assert isinstance(self._ops, RemoteOps)

        batch = self._get_batch()
        for _ in range(self._params.grad_iters):
            self._ops.update_critic_with_grad(await self._ops.get_critic_grad(batch))

        for _ in range(self._params.grad_iters):
            if self._ops.update_actor_with_grad(await self._ops.get_actor_grad(batch)):  # early stop
                break
