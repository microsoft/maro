from typing import Callable, Dict

import numpy as np
import torch

from maro.rl.utils import discount_cumsum
from maro.rl_v3.model import VNet
from maro.rl_v3.policy import DiscretePolicyGradient, RLPolicy
from maro.rl_v3.replay_memory import FIFOReplayMemory
from maro.rl_v3.utils import TransitionBatch, ndarray_to_tensor

from .abs_trainer import SingleTrainer
from .train_worker import SingleTrainWorker


class DiscreteActorCriticWorker(SingleTrainWorker):
    def __init__(
        self,
        name: str,
        device: torch.device,
        get_v_critic_net_func: Callable[[], VNet],
        reward_discount: float = 0.9,
        critic_loss_coef: float = 0.1,
        critic_loss_cls: Callable = None,
        clip_ratio: float = None,
        lam: float = 0.9,
        min_logp: float = None,
        enable_data_parallelism: bool = False
    ) -> None:
        super(DiscreteActorCriticWorker, self).__init__(name, device, enable_data_parallelism)

        self._get_v_critic_net_func = get_v_critic_net_func
        self._reward_discount = reward_discount
        self._critic_loss_coef = critic_loss_coef
        self._critic_loss_func = critic_loss_cls() if critic_loss_cls is not None else torch.nn.MSELoss()
        self._clip_ratio = clip_ratio
        self._lam = lam
        self._min_logp = min_logp

    def _register_policy_impl(self, policy: RLPolicy) -> None:
        assert isinstance(policy, DiscretePolicyGradient)

        self._policy = policy
        self._v_critic_net = self._get_v_critic_net_func()
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

    def _get_critic_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        states = ndarray_to_tensor(batch.states, self._device)  # s

        self._policy.train()
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
            self._policy.eval()
            logps_old = self._policy.get_state_action_logps(states, actions)
        else:
            logps_old = None

        state_values = self._v_critic_net.v_values(states)
        values = state_values.detach().numpy()
        values = np.concatenate([values, values[-1:]])
        rewards = np.concatenate([batch.rewards, values[-1:]])

        deltas = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]  # r + gamma * v(s') - v(s)
        advantages = ndarray_to_tensor(discount_cumsum(deltas, self._reward_discount * self._lam), self._device)

        action_probs = self._policy.get_action_probs(states)
        logps = torch.log(action_probs.gather(1, actions).squeeze())
        logps = torch.clamp(logps, min=self._min_logp, max=.0)
        if self._clip_ratio is not None:
            ratio = torch.exp(logps - logps_old)
            clipped_ratio = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
            actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
        else:
            actor_loss = -(logps * advantages).mean()  # I * delta * log pi(a|s)

        return self._policy.get_gradients(actor_loss)

    def update(self) -> None:
        """
        Reference: https://tinyurl.com/2ezte4cr
        """
        grad_dict = self._get_batch_grad(self._batch, scope="all")
        self._policy.train()
        self._policy.apply_gradients(grad_dict["actor_grad"])
        self._v_critic_net.train()
        self._v_critic_net.apply_gradients(grad_dict["critic_grad"])

    def get_worker_state_dict(self, scope: str = "all") -> dict:
        ret_dict = {}
        if scope in ("all", "actor"):
            ret_dict["policy_state"] = self._policy.get_policy_state()
        if scope in ("all", "critic"):
            ret_dict["critic_state"] = self._v_critic_net.get_net_state()
        return ret_dict

    def set_worker_state_dict(self, worker_state_dict: dict, scope: str = "all") -> None:
        if scope in ("all", "actor"):
            self._policy.set_policy_state(worker_state_dict["policy_state"])
        if scope in ("all", "critic"):
            self._v_critic_net.set_net_state(worker_state_dict["critic_state"])


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
        get_v_critic_net_func: Callable[[], VNet],
        policy: DiscretePolicyGradient = None,
        replay_memory_capacity: int = 10000,
        train_batch_size: int = 128,
        grad_iters: int = 1,
        reward_discount: float = 0.9,
        lam: float = 0.9,
        clip_ratio: float = None,
        critic_loss_cls: Callable = None,
        min_logp: float = None,
        critic_loss_coef: float = 0.1,
        device: str = None,
        enable_data_parallelism: bool = False
    ) -> None:
        super(DiscreteActorCritic, self).__init__(name, device, enable_data_parallelism)

        self._replay_memory_capacity = replay_memory_capacity

        self._get_v_critic_net_func = get_v_critic_net_func
        if policy is not None:
            self.register_policy(policy)

        self._train_batch_size = train_batch_size
        self._reward_discount = reward_discount
        self._lam = lam
        self._clip_ratio = clip_ratio
        self._min_logp = min_logp
        self._grad_iters = grad_iters
        self._critic_loss_coef = critic_loss_coef
        self._critic_loss_cls = critic_loss_cls

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def _register_policy_impl(self, policy: DiscretePolicyGradient) -> None:
        self._worker = DiscreteActorCriticWorker(
            name="worker", device=self._device, get_v_critic_net_func=self._get_v_critic_net_func,
            reward_discount=self._reward_discount, critic_loss_coef=self._critic_loss_coef,
            critic_loss_cls=self._critic_loss_cls, clip_ratio=self._clip_ratio, lam=self._lam,
            min_logp=self._min_logp, enable_data_parallelism=self._enable_data_parallelism
        )
        self._worker.register_policy(policy)

        self._replay_memory = FIFOReplayMemory(
            capacity=self._replay_memory_capacity, state_dim=policy.state_dim,
            action_dim=policy.action_dim
        )

    def _train_step_impl(self) -> None:
        self._worker.set_batch(self._get_batch())
        for _ in range(self._grad_iters):
            self._worker.update()

    def get_policy_state_dict(self) -> Dict[str, object]:
        return {self._policy_name: self._worker.get_policy_state()}

    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        assert len(policy_state_dict) == 1 and self._policy_name in policy_state_dict
        self._worker.set_policy_state(list(policy_state_dict.values())[0])
