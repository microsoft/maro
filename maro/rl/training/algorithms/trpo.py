# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Callable, Dict, Optional, cast

import numpy as np
import torch

from maro.rl.model import VNet
from maro.rl.policy import DiscretePolicyGradient, ContinuousRLPolicy, RLPolicy
from maro.rl.training import AbsTrainOps, AbsTrainer, BaseTrainerParams, FIFOReplayMemory, remote
from maro.rl.utils import TransitionBatch, get_torch_device, discount_cumsum, ndarray_to_tensor

# trpo_base needs to be tuned in [TRPOOps, TRPOTrainer] to fully fit MARO style.

# batch 更新权重或者数据量
@dataclass
class TRPOParams(BaseTrainerParams):
    """Refer to Spinning Up params

    ac_kwargs (dict): Any kwargs appropriate for the actor_critic
        function you provided to TRPO.
    seed (int): Seed for random number generators.
    steps_per_epoch (int): Number of steps of interaction (state-action pairs)
        for the agent and the environment in each epoch.
    epochs (int): Number of epochs of interaction (equivalent to
        number of policy updates) to perform.
    gamma (float): Discount factor. (Always between 0 and 1.)
    delta (float): KL-divergence limit for TRPO / NPG update.
        (Should be small for stability. Values like 0.01, 0.05.)
    vf_lr (float): Learning rate for value function optimizer.
    train_v_iters (int): Number of gradient descent steps to take on
        value function per epoch.
    damping_coeff (float): Artifact for numerical stability, should be
        smallish. Adjusts Hessian-vector product calculation:
        .. math:: Hv \\rightarrow (\\alpha I + H)v
        where :math:`\\alpha` is the damping coefficient.
        Probably don't play with this hyperparameter.
    cg_iters (int): Number of iterations of conjugate gradient to perform.
        Increasing this will lead to a more accurate approximation
        to :math:`H^{-1} g`, and possibly slightly-improved performance,
        but at the cost of slowing things down.
        Also probably don't play with this hyperparameter.
    backtrack_iters (int): Maximum number of steps allowed in the
        backtracking line search. Since the line search usually doesn't
        backtrack, and usually only steps back once when it does, this
        hyperparameter doesn't often matter.
    backtrack_coeff (float): How far back to step during backtracking line
        search. (Always between 0 and 1, usually above 0.5.)
    lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
        close to 1.)
    max_ep_len (int): Maximum length of trajectory / episode / rollout.
    logger_kwargs (dict): Keyword args for EpochLogger.
    save_freq (int): How often (in terms of gap between epochs) to save
        the current policy and value function.
    algo: 'trpo' algo.
    """

    # TODO:need to determine which parameters are required

    # maro common params
    get_v_critic_net_func: Callable[[], VNet]
    critic_loss_cls: Optional[Callable] = None
    grad_iters: int = 1
    num_epochs: int = 1
    update_target_every: int = 5
    random_overwrite: bool = False

    # Refer to Spinning Up params,
    ac_kwargs: dict = dict()
    seed: int = 0  # seed
    # steps_per_epoch: int = 4000
    # epochs: int = 50
    gamma: float = 0.99  # _reward_discount
    delta: float = 0.01  # max kl
    # vf_lr: float = 1e-3
    # train_v_iters: int = 80
    damping_coeff: float = 0.1
    # cg_iters: int = 10
    # backtrack_iters: int = 10
    # backtrack_coeff: float = 0.8
    lam: float = 0.97  # GAE
    # max_ep_len: int = 1000
    # logger_kwargs: dict = dict()
    # save_freq: int = 10
    # algo = 'trpo'


class TRPOOps(AbsTrainOps):
    def __init__(
        self,
        name: str,
        policy: RLPolicy,
        params: TRPOParams,
        reward_discount: float = 0.9,  # gamma
        parallelism: int = 1,
    ) -> None:
        super(TRPOOps, self).__init__(
            name=name,
            policy=policy,
            parallelism=parallelism,
        )

        # TRPO can be used for environments with either discrete or continuous action spaces.
        assert isinstance(self._policy, (ContinuousRLPolicy, DiscretePolicyGradient))

        self._reward_discount = reward_discount  # gamma
        self._v_critic_net = params.get_v_critic_net_func()
        self._critic_loss_func = (
            params.critic_loss_cls() if params.critic_loss_cls is not None else torch.nn.MSELoss()
        )  # torch.nn.MSELoss() used to calculate loss between predict and target
        self._is_discrete_action = isinstance(self._policy, DiscretePolicyGradient)

        # TODO: Split train_step() to ops._get_critic_loss and ....

    def _get_critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """Compute the critic loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The critic loss of the batch.
        """
        # TODO: important!!! (maybe)not done. refer to maro-ppo
        assert isinstance(batch, TransitionBatch)

        self._v_critic_net.train()
        states = ndarray_to_tensor(batch.states, self._device)  # states
        state_values = self._v_critic_net.v_values(states)  # .v_values(states)

        values = state_values.cpu().detach().numpy()  # get_flat_grad_from VNet
        values = np.concatenate([values[1:], values[-1:]])  # TODO:why? no 1st, double last
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
        # TODO: mainly to do !!!
        assert isinstance(self._policy, (ContinuousRLPolicy, DiscretePolicyGradient))
        self._policy.train()
        states = ndarray_to_tensor(batch.states, device=self._device)  # states
        actions = ndarray_to_tensor(batch.actions, device=self._device)
        advantages = ndarray_to_tensor(batch.advantages, device=self._device)
        logp_old = ndarray_to_tensor(batch.old_logps, device=self._device)
        if self._is_discrete_action:
            actions = actions.long()
        logp = self._policy.get_states_actions_logps(states, actions)
        ratio = torch.exp(logp - logp_old)  # pi(a|s) / pi_old(a|s)
        actor_loss = -torch.mean(ratio * advantages)  # -(ratio * advantages).mean()
        return actor_loss

    @remote
    def get_actor_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
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

    def build(self) -> None:
        self._ops = cast(TRPOOps, self.get_ops())
        self._replay_memory = (
            FIFOReplayMemory(  # TODO: memory design. which maro-memory is needed? need sample() func in _get_batch
                capacity=self._replay_memory_capacity,
                state_dim=self._ops.policy_state_dim,
                action_dim=self._ops.policy_action_dim,
                random_overwrite=self._params.random_overwrite,
            )
        )

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
