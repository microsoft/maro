# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import collections
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from maro.rl_v3.learning import ExpElement
from maro.rl_v3.model import VNet
from maro.rl_v3.policy import DiscretePolicyGradient
from maro.rl_v3.training import AbsTrainOps, FIFOReplayMemory, SingleTrainer, TrainerParams
from maro.rl_v3.utils import AbsTransitionBatch, TransitionBatch, discount_cumsum, ndarray_to_tensor


@dataclass
class DiscreteActorCriticParams(TrainerParams):
    """
    get_v_critic_net_func (Callable[[], VNet]): Function to get V critic net.
    reward_discount (float): Reward decay as defined in standard RL terminology. Defaults to 0.9.
    grad_iters (int): Number of iterations to calculate gradients. Defaults to 1.
    critic_loss_coef (float): Coefficient for critic loss in total loss. Defaults to 0.1.
    critic_loss_cls (Callable): Loss function. Defaults to "mse".
    clip_ratio (float): Clip ratio in the PPO algorithm (https://arxiv.org/pdf/1707.06347.pdf). Defaults to None,
        in which case the actor loss is calculated using the usual policy gradient theorem.
    lam (float): Lambda value for generalized advantage estimation (TD-Lambda). Defaults to 0.9.
    min_logp (float): Lower bound for clamping logP values during learning. This is to prevent logP from becoming
        very large in magnitude and causing stability issues. Defaults to None, which means no lower bound.
    """
    get_v_critic_net_func: Callable[[], VNet] = None
    reward_discount: float = 0.9
    grad_iters: int = 1
    critic_loss_coef: float = 0.1
    critic_loss_cls: Callable = None
    clip_ratio: float = None
    lam: float = 0.9
    min_logp: Optional[float] = None

    def __post_init__(self) -> None:
        assert self.get_v_critic_net_func is not None

    def extract_ops_params(self) -> Dict[str, object]:
        return {
            "device": self.device,
            "enable_data_parallelism": self.enable_data_parallelism,
            "get_v_critic_net_func": self.get_v_critic_net_func,
            "reward_discount": self.reward_discount,
            "critic_loss_coef": self.critic_loss_coef,
            "critic_loss_cls": self.critic_loss_cls,
            "clip_ratio": self.clip_ratio,
            "lam": self.lam,
            "min_logp": self.min_logp,
        }


class DiscreteActorCriticOps(AbsTrainOps):
    def __init__(
        self,
        device: str,
        get_policy_func: Callable[[], DiscretePolicyGradient],
        get_v_critic_net_func: Callable[[], VNet],
        enable_data_parallelism: bool = False,
        *,
        reward_discount: float = 0.9,
        critic_loss_coef: float = 0.1,
        critic_loss_cls: Callable = None,
        clip_ratio: float = None,
        lam: float = 0.9,
        min_logp: float = None,
    ) -> None:
        super(DiscreteActorCriticOps, self).__init__(
            device=device,
            is_single_scenario=True,
            get_policy_func=get_policy_func,
            enable_data_parallelism=enable_data_parallelism
        )

        assert isinstance(self._policy, DiscretePolicyGradient)

        self._reward_discount = reward_discount
        self._critic_loss_coef = critic_loss_coef
        self._critic_loss_func = critic_loss_cls() if critic_loss_cls is not None else torch.nn.MSELoss()
        self._clip_ratio = clip_ratio
        self._lam = lam
        self._min_logp = min_logp
        self._v_critic_net = get_v_critic_net_func()
        self._v_critic_net.to(self._device)

    def get_batch_grad(
        self,
        batch: TransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Reference: https://tinyurl.com/2ezte4cr
        """
        assert scope in ("all", "actor", "critic"), \
            f"Unrecognized scope {scope}. Excepting 'all', 'actor', or 'critic'."

        grad_dict = {}
        if scope in ("all", "actor"):
            grad_dict["actor_grad"] = self._get_actor_grad(batch)

        if scope in ("all", "critic"):
            grad_dict["critic_grad"] = self._get_critic_grad(batch)

        return grad_dict

    def _dispatch_batch(self, batch: AbsTransitionBatch, num_sub_batches: int) -> List[AbsTransitionBatch]:
        raise NotImplementedError

    def _dispatch_tensor_dict(self, tensor_dict: Dict[str, object], num_sub_batches: int) -> List[Dict[str, object]]:
        raise NotImplementedError

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
        assert isinstance(self._policy, DiscretePolicyGradient)

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

    def update(self, grad_iters: int) -> None:
        """Reference: https://tinyurl.com/2ezte4cr
        """
        for _ in range(grad_iters):
            grad_dict = self._get_batch_grad(self._batch, scope="all")
            self._policy.train()
            self._policy.apply_gradients(grad_dict["actor_grad"])
            self._v_critic_net.train()
            self._v_critic_net.apply_gradients(grad_dict["critic_grad"])

    def get_state_dict(self, scope: str = "all") -> dict:
        ret_dict = {}
        if scope in ("all", "actor"):
            ret_dict["policy_state"] = self._policy.get_state()
        if scope in ("all", "critic"):
            ret_dict["critic_state"] = self._v_critic_net.get_net_state()
        return ret_dict

    def set_state_dict(self, ops_state_dict: dict, scope: str = "all") -> None:
        if scope in ("all", "actor"):
            self._policy.set_state(ops_state_dict["policy_state"])
        if scope in ("all", "critic"):
            self._v_critic_net.set_net_state(ops_state_dict["critic_state"])


class DiscreteActorCritic(SingleTrainer):
    """Actor Critic algorithm with separate policy and value models.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.
        https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
    """
    def __init__(self, name: str, params: DiscreteActorCriticParams) -> None:
        super(DiscreteActorCritic, self).__init__(name, params)
        self._params = params
        self._ops_params = {}
        self._ops_name = f"{self._name}.ops"

        self._replay_memory_dict = {}

    def build(self) -> None:
        self._ops_params = {
            "get_policy_func": self._get_policy_func,
            **self._params.extract_ops_params(),
        }
        self._ops = self.get_ops(self._ops_name)

        self._replay_memory_dict = collections.defaultdict(lambda: FIFOReplayMemory(
            capacity=self._params.replay_memory_capacity,
            state_dim=self._ops.policy_state_dim,
            action_dim=self._ops.policy_action_dim
        ))

    def record(self, exp_element: ExpElement) -> None:
        for agent_name in exp_element.agent_names:
            memory = self._replay_memory_dict[agent_name]
            transition_batch = TransitionBatch(
                states=np.expand_dims(exp_element.agent_state_dict[agent_name], axis=0),
                actions=np.expand_dims(exp_element.action_dict[agent_name], axis=0),
                rewards=np.array([exp_element.reward_dict[agent_name]]),
                terminals=np.array([exp_element.terminal_dict[agent_name]]),
                next_states=np.expand_dims(
                    exp_element.next_agent_state_dict.get(agent_name, exp_element.agent_state_dict[agent_name]),
                    axis=0,
                ),
            )
            memory.put(transition_batch)

    def _get_local_ops_by_name(self, ops_name: str) -> AbsTrainOps:
        return DiscreteActorCriticOps(**self._ops_params)

    def _get_batch(self, agent_name: str, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory_dict[agent_name].sample(batch_size if batch_size is not None else self._batch_size)

    async def train_step(self):
        for agent_name in self._replay_memory_dict:
            await asyncio.gather(self._ops.set_batch(self._get_batch(agent_name)))
            await asyncio.gather(self._ops.update(self._params.grad_iters))
