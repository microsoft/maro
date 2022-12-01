# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, cast
import argparse,scipy.optimize
from torch.autograd import Variable

from maro.rl.model import VNet
from maro.rl.policy import ContinuousRLPolicy, DiscretePolicyGradient, RLPolicy
from maro.rl.training import AbsTrainOps, BaseTrainerParams, FIFOReplayMemory, RemoteOps, SingleAgentTrainer, remote
from maro.rl.utils import TransitionBatch, discount_cumsum, get_torch_device, ndarray_to_tensor

from maro.rl.training.algorithms.trpo_base.trpo_model import *
import math

import numpy as np

import torch


@dataclass
class TRPOParams(BaseTrainerParams):
    """
    get_q_critic_net_func (Callable[[], QNet]): Function to get Q critic net.
    num_epochs (int, default=1): Number of training epochs per call to ``learn``.
    update_target_every (int, default=5): Number of training rounds between policy target model updates.
    q_value_loss_cls (str, default=None): A string indicating a loss class provided by torch.nn or a custom
        loss class for the Q-value loss. If it is a string, it must be a key in ``TORCH_LOSS``.
        If it is None, use MSE.
    soft_update_coef (float, default=1.0): Soft update coefficient, e.g.,
        target_model = (soft_update_coef) * eval_model + (1-soft_update_coef) * target_model.
    random_overwrite (bool, default=False): This specifies overwrite behavior when the replay memory capacity
        is reached. If True, overwrite positions will be selected randomly. Otherwise, overwrites will occur
        sequentially with wrap-around.
    min_num_to_trigger_training (int, default=0): Minimum number required to start training.
    """

    get_v_critic_net_func: Callable[[], VNet]
    grad_iters: int = 1
    critic_loss_cls: Optional[Callable] = None
    lam: float = 0.9
    min_logp: float = float("-inf")
    clip_ratio: Optional[float] = None


class TRPOOps(AbsTrainOps):
    """DDPG algorithm implementation. Reference: https://spinningup.openai.com/en/latest/algorithms/ddpg.html"""

    def __init__(
        self,
        name: str,
        policy: RLPolicy,
        params: TRPOParams,
        reward_discount: float = 0.9,
        parallelism: int = 1,
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
        self._lam = params.lam
        self._min_logp = params.min_logp
        self._v_critic_net = params.get_v_critic_net_func()
        self._is_discrete_action = isinstance(self._policy, DiscretePolicyGradient)
        self.parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
        self.parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                                 help='discount factor (default: 0.995)')
        self.parser.add_argument('--env-name', default="Reacher-v4", metavar='G',
                                 help='name of the environment to run')
        self.parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                                 help='gae (default: 0.97)')
        self.parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                                 help='l2 regularization regression (default: 1e-3)')
        self.parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                                 help='max kl value (default: 1e-2)')
        self.parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                                 help='damping (default: 1e-1)')
        self.parser.add_argument('--seed', type=int, default=543, metavar='N',
                                 help='random seed (default: 1)')
        self.parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                                 help='random seed (default: 1)')
        self.parser.add_argument('--render', action='store_true',
                                 help='render the environment')
        self.parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                                 help='interval between training status logs (default: 10)')
        self.args = self.parser.parse_args()
        self.policy_net = Policy(171, 1)
        self.value_net = Value(171)

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

    def set_flat_params_to(self,model, flat_params):
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size

    def normal_log_density(self,x, mean, log_std, std):
        var = std.pow(2)
        # linshi = -(x - mean).pow(2) / (2 * var)

        log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
        return log_density.sum(1, keepdim=True)

    def get_flat_params_from(self,model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))

        flat_params = torch.cat(params)
        return flat_params


    def get_flat_grad_from(self,net, grad_grad=False):
        grads = []
        for param in net.parameters():
            if grad_grad:
                grads.append(param.grad.grad.view(-1))
            else:
                grads.append(param.grad.view(-1))

        flat_grad = torch.cat(grads)
        return flat_grad

    def conjugate_gradients(self,Avp, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = Avp(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def linesearch(self,model,f,x,fullstep,expected_improve_rate,max_backtracks=10,accept_ratio=.1):
        fval = f(True).data
        # print("fval before", fval.item())
        for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            self.set_flat_params_to(model, xnew)
            newfval = f(True).data
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            # print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                # print("fval after", newfval.item())
                return True, xnew
        return False, x


    def get_loss(self,volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = self.policy_net(Variable(self.states))
        else:
            action_means, action_log_stds, action_stds = self.policy_net(Variable(self.states))

        log_prob = self.normal_log_density(Variable(self.actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(self.advantages) * torch.exp(log_prob - Variable(self.fixed_log_prob))
        return action_loss.mean()

    def get_kl(self):
        mean1, log_std1, std1 = self.policy_net(Variable(self.states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def trpo_step(self,model, get_loss, get_kl, max_kl, damping):
        loss = get_loss()
        grads = torch.autograd.grad(loss, model.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        def Fvp(v):
            kl = get_kl()
            kl = kl.mean()

            grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, model.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + v * damping

        stepdir = self.conjugate_gradients(Fvp, -loss_grad, 10)

        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / max_kl)
        fullstep = stepdir / lm[0]

        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        # print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

        prev_params = self.get_flat_params_from(model)
        success, new_params = self.linesearch(model, get_loss, prev_params, fullstep,
                                         neggdotstepdir / lm[0])
        self.set_flat_params_to(model, new_params)

        return loss


    def get_value_loss(self,flat_params):
        self.set_flat_params_to(self.value_net, torch.Tensor(flat_params))
        for param in self.value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_ = self.value_net(Variable(self.states))
        value_loss = (values_ - self.targets).pow(2).mean()
        for param in self.value_net.parameters():
            value_loss += param.pow(2).sum() * self.args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), self.get_flat_grad_from(self.value_net).data.double().numpy())


    def _get_actor_loss(self, batch: TransitionBatch):
        assert isinstance(self._policy, DiscretePolicyGradient) or isinstance(self._policy, ContinuousRLPolicy)
        self._policy.train()
        self.rewards = ndarray_to_tensor(batch.rewards)
        self.actions = ndarray_to_tensor(batch.actions)
        self.states = ndarray_to_tensor(batch.states)
        self.returns = ndarray_to_tensor(batch.returns)
        self.deltas = torch.Tensor(self.actions.size(0), 1)
        self.values = self.value_net(Variable(self.states))

        advantages = torch.Tensor(batch.advantages)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(self.rewards.size(0))):
            # returns[i] = rewards[i] + self.args.gamma * prev_return * masks[i]
            # deltas[i] = rewards[i] + self.args.gamma * prev_value * masks[i] - values.data[i]
            # advantages[i] = deltas[i] + self.args.gamma * self.args.tau * prev_advantage * masks[i]
            self.returns[i] = self.rewards[i] + self.args.gamma * prev_return
            self.deltas[i] = self.rewards[i] + self.args.gamma * prev_value - self.values.data[i]
            advantages[i] = self.deltas[i] + self.args.gamma * self.args.tau * prev_advantage

            prev_return = self.returns[i]
            prev_value = self.values.data[i]
            prev_advantage = advantages[i]
        self.targets = Variable(self.returns)

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(self.get_value_loss,
                                                                self.get_flat_params_from(self.value_net).double().numpy(),
                                                                maxiter=25)
        self.set_flat_params_to(self.value_net, torch.Tensor(flat_params))
        self.advantages = (advantages - advantages.mean()) / advantages.std()
        action_means, action_log_stds, action_stds = self.policy_net(Variable(self.states))
        self.fixed_log_prob = self.normal_log_density(Variable(self.actions), action_means, action_log_stds, action_stds).data.clone()
        actor_loss = self.trpo_step(self.policy_net, self.get_loss, self.get_kl, self.args.max_kl, self.args.damping)



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
        loss, early_stop = self._get_actor_loss(batch)
        return self._policy.get_gradients(loss), early_stop

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


class TRPOTrainer(SingleAgentTrainer):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        https://arxiv.org/pdf/1509.02971.pdf
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg
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
        )

    def _get_batch(self) -> TransitionBatch:
        batch = self._replay_memory.sample(-1)

        batch.advantages = (batch.advantages - batch.advantages.mean()) / batch.advantages.std()

        return batch

    def train_step(self) -> None:
        assert isinstance(self._ops, TRPOOps)
        #  mask -> actions个1
        batch = self._get_batch()
        # trpo_main.update_params(batch)
        for _ in range(self._params.grad_iters):
            self._ops.update_critic(batch)
            self._ops.update_actor(batch)

        # for _ in range(self._params.grad_iters):
        #     self._ops.update_critic(batch)
        #     self._ops.update_actor(batch)

        # for _ in range(self._params.grad_iters):
        #     early_stop = self._ops.update_actor(batch)
        #     if early_stop:
        #         break

    async def train_step_as_task(self) -> None:
        assert isinstance(self._ops, RemoteOps)

        batch = self._get_batch()
        for _ in range(self._params.grad_iters):
            self._ops.update_critic_with_grad(await self._ops.get_critic_grad(batch))

        for _ in range(self._params.grad_iters):
            if self._ops.update_actor_with_grad(await self._ops.get_actor_grad(batch)):  # early stop
                break
