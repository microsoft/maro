import asyncio
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from maro.rl.utils import discount_cumsum
from maro.rl_v3.model import VNet
from maro.rl_v3.policy import DiscretePolicyGradient
from maro.rl_v3.replay_memory import FIFOReplayMemory
from maro.rl_v3.utils import TransitionBatch, ndarray_to_tensor

from .abs_train_ops import SingleTrainOps
from .abs_trainer import SingleTrainer


class DiscreteActorCriticOps(SingleTrainOps):
    def __init__(
        self,
        policy: DiscretePolicyGradient,
        v_critic_net: VNet,
        device: torch.device,
        enable_data_parallelism: bool = False,
        *,
        reward_discount: float = 0.9,
        critic_loss_coef: float = 0.1,
        critic_loss_cls: Callable = None,
        clip_ratio: float = None,
        lam: float = 0.9,
        min_logp: float = None,
    ) -> None:
        assert isinstance(policy, DiscretePolicyGradient)
        super(DiscreteActorCriticOps, self).__init__(device, enable_data_parallelism)

        self.policy = policy
        self._reward_discount = reward_discount
        self._critic_loss_coef = critic_loss_coef
        self._critic_loss_func = critic_loss_cls() if critic_loss_cls is not None else torch.nn.MSELoss()
        self._clip_ratio = clip_ratio
        self._lam = lam
        self._min_logp = min_logp
        self._v_critic_net = v_critic_net
        self._v_critic_net.to(self._device)

    def get_batch_grad(
        self,
        batch: TransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Reference: https://tinyurl.com/2ezte4cr
        """
        assert scope in ("all", "actor", "critic"), \
            f"Unrecognized scope {scope}. Excepting 'all', 'actor', or 'critic'."

        grad_dict = {}
        if scope in ("all", "actor"):
            grad_dict["actor_grad"] = self._get_actor_grad(batch)

        if scope in ("all", "critic"):
            grad_dict["critic_grad"] = self._get_critic_grad(batch)

        return grad_dict

    def _dispatch_batch(self, batch: TransitionBatch, num_ops: int) -> List[TransitionBatch]:
        raise NotImplementedError

    def _dispatch_tensor_dict(self, tensor_dict: Dict[str, object], num_ops: int) -> List[Dict[str, object]]:
        raise NotImplementedError

    def _get_critic_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        states = ndarray_to_tensor(batch.states, self._device)  # s

        self.policy.train()
        self._v_critic_net.train()

        state_values = self._v_critic_net.v_values(states)
        values = state_values.detach().numpy()
        values = np.concatenate([values, values[-1:]])
        rewards = np.concatenate([batch.rewards, values[-1:]])

        returns = ndarray_to_tensor(discount_cumsum(rewards, self._reward_discount)[:-1], self._device)
        critic_loss = self._critic_loss_func(state_values, returns)

        return self._v_critic_net.get_gradients(critic_loss * self._critic_loss_coef)

    def _get_actor_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        states = ndarray_to_tensor(batch.states, self._device)  # s
        actions = ndarray_to_tensor(batch.actions, self._device).long()  # a

        if self._clip_ratio is not None:
            self.policy.eval()
            logps_old = self.policy.get_state_action_logps(states, actions)
        else:
            logps_old = None

        state_values = self._v_critic_net.v_values(states)
        values = state_values.detach().numpy()
        values = np.concatenate([values, values[-1:]])
        rewards = np.concatenate([batch.rewards, values[-1:]])

        deltas = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]  # r + gamma * v(s') - v(s)
        advantages = ndarray_to_tensor(discount_cumsum(deltas, self._reward_discount * self._lam), self._device)

        action_probs = self.policy.get_action_probs(states)
        logps = torch.log(action_probs.gather(1, actions).squeeze())
        logps = torch.clamp(logps, min=self._min_logp, max=.0)
        if self._clip_ratio is not None:
            ratio = torch.exp(logps - logps_old)
            clipped_ratio = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
            actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
        else:
            actor_loss = -(logps * advantages).mean()  # I * delta * log pi(a|s)

        return self.policy.get_gradients(actor_loss)

    def update(self) -> None:
        """
        Reference: https://tinyurl.com/2ezte4cr
        """
        grad_dict = self._get_batch_grad(self._batch, scope="all")
        self.policy.train()
        self.policy.apply_gradients(grad_dict["actor_grad"])
        self._v_critic_net.train()
        self._v_critic_net.apply_gradients(grad_dict["critic_grad"])

    def get_ops_state_dict(self, scope: str = "all") -> dict:
        ret_dict = {}
        if scope in ("all", "actor"):
            ret_dict["policy_state"] = self.policy.get_policy_state()
        if scope in ("all", "critic"):
            ret_dict["critic_state"] = self._v_critic_net.get_net_state()
        return ret_dict

    def set_ops_state_dict(self, ops_state_dict: dict, scope: str = "all") -> None:
        if scope in ("all", "actor"):
            self.policy.set_policy_state(ops_state_dict["policy_state"])
        if scope in ("all", "critic"):
            self._v_critic_net.set_net_state(ops_state_dict["critic_state"])


class DiscreteActorCritic(SingleTrainer):
    """Actor Critic algorithm with separate policy and value models.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.
        https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

    Args:
        name (str): Unique identifier for the policy.
        get_v_critic_net_func (Callable[[], VNet]): Function to get V critic net.
        policy (DiscretePolicyGradient): The policy to be trained.
        replay_memory_capacity (int): Capacity of the replay memory. Defaults to 10000.
        train_batch_size (int): Batch size for training the Q-net. Defaults to 128.
        grad_iters (int): Number of iterations to calculate gradients. Defaults to 1.
        reward_discount (float): Reward decay as defined in standard RL terminology. Defaults to 0.9.
        lam (float): Lambda value for generalized advantage estimation (TD-Lambda). Defaults to 0.9.
        clip_ratio (float): Clip ratio in the PPO algorithm (https://arxiv.org/pdf/1707.06347.pdf). Defaults to None,
            in which case the actor loss is calculated using the usual policy gradient theorem.
        critic_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for computing
            the critic loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        min_logp (float): Lower bound for clamping logP values during learning. This is to prevent logP from becoming
            very large in magnitude and causing stability issues. Defaults to None, which means no lower bound.
        critic_loss_coef (float): Coefficient for critic loss in total loss. Defaults to 1.0.
        device (str): Identifier for the torch device. The policy will be moved to the specified device. If it is
            None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise. Defaults to None.
        enable_data_parallelism (bool): Whether to enable data parallelism in this trainer. Defaults to False.
    """
    def __init__(
        self,
        name: str,
        ops_creator: Dict[str, Callable],
        dispatcher_address: Tuple[str, int] = None,
        *,
        state_dim: int,
        action_dim: int,
        replay_memory_size: int = 10000,
        train_batch_size: int = 128,
        grad_iters: int = 1,
        device: str = None,
        enable_data_parallelism: bool = False
    ) -> None:
        super(DiscreteActorCritic, self).__init__(
            name, ops_creator,
            dispatcher_address=dispatcher_address,
            device=device,
            enable_data_parallelism=enable_data_parallelism,
            train_batch_size=train_batch_size
        )

        self._grad_iters = grad_iters
        self._replay_memory = FIFOReplayMemory(replay_memory_size, state_dim, action_dim)

    async def train_step(self):
        self._ops.set_batch(self._get_batch())
        for _ in range(self._grad_iters):
            await asyncio.gather(self._ops.update())
