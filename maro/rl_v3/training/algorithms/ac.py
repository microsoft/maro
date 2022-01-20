# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import collections
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from maro.rl_v3.model import VNet
from maro.rl_v3.policy import DiscretePolicyGradient
from maro.rl_v3.rollout import ExpElement
from maro.rl_v3.training import AbsTrainOps, FIFOReplayMemory, RemoteOps, SingleTrainer, TrainerParams, remote
from maro.rl_v3.utils import (
    TransitionBatch, average_grads, discount_cumsum, merge_transition_batches, ndarray_to_tensor
)


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
    data_parallelism: int = 1

    def __post_init__(self) -> None:
        assert self.get_v_critic_net_func is not None

    def extract_ops_params(self) -> Dict[str, object]:
        return {
            "device": self.device,
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
        *,
        reward_discount: float = 0.9,
        critic_loss_coef: float = 0.1,
        critic_loss_cls: Callable = None,
        clip_ratio: float = None,
        lam: float = 0.9,
        min_logp: float = None
    ) -> None:
        super(DiscreteActorCriticOps, self).__init__(
            device=device,
            is_single_scenario=True,
            get_policy_func=get_policy_func
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

    @remote
    def get_critic_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        self._v_critic_net.train()

        states = ndarray_to_tensor(batch.states, self._device)  # s
        state_values = self._v_critic_net.v_values(states)
        returns = ndarray_to_tensor(batch.returns, self._device)
        critic_loss = self._critic_loss_func(state_values, returns)

        return self._v_critic_net.get_gradients(critic_loss * self._critic_loss_coef)

    @remote
    def get_actor_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        assert isinstance(self._policy, DiscretePolicyGradient)
        self._policy.train()

        states = ndarray_to_tensor(batch.states, self._device)  # s
        actions = ndarray_to_tensor(batch.actions, self._device).long()  # a
        advantages = ndarray_to_tensor(batch.advantages, self._device)

        if self._clip_ratio is not None:
            self._policy.eval()
            logps_old = self._policy.get_state_action_logps(states, actions)
        else:
            logps_old = None

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

    def update_critic(self, grad_dict: dict) -> None:
        """Reference: https://tinyurl.com/2ezte4cr
        """
        self._v_critic_net.train()
        self._v_critic_net.apply_gradients(grad_dict)

    def update_actor(self, grad_dict: dict) -> None:
        """Reference: https://tinyurl.com/2ezte4cr
        """
        self._policy.train()
        self._policy.apply_gradients(grad_dict)

    def get_state(self, scope: str = "all") -> dict:
        ret_dict = {}
        if scope in ("all", "actor"):
            ret_dict["policy_state"] = self._policy.get_state()
        if scope in ("all", "critic"):
            ret_dict["critic_state"] = self._v_critic_net.get_net_state()
        return ret_dict

    def set_state(self, ops_state_dict: dict, scope: str = "all") -> None:
        if scope in ("all", "actor"):
            self._policy.set_state(ops_state_dict["policy_state"])
        if scope in ("all", "critic"):
            self._v_critic_net.set_net_state(ops_state_dict["critic_state"])

    def preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        assert self._is_valid_transition_batch(batch)
        # Preprocess returns
        batch.calc_returns(self._reward_discount)

        # Preprocess advantages
        states = ndarray_to_tensor(batch.states, self._device)  # s
        state_values = self._v_critic_net.v_values(states)
        values = state_values.detach().numpy()
        values = np.concatenate([values, values[-1:]])
        rewards = np.concatenate([batch.rewards, values[-1:]])
        deltas = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]  # r + gamma * v(s') - v(s)
        advantages = discount_cumsum(deltas, self._reward_discount * self._lam)
        batch.advantages = advantages
        return batch


class DiscreteActorCritic(SingleTrainer):
    """Actor Critic algorithm with separate policy and value models.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.
        https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
    """
    def __init__(self, name: str, params: DiscreteActorCriticParams) -> None:
        super(DiscreteActorCritic, self).__init__(name, params)
        self._params = params
        self._ops_name = f"{self._name}.ops"
        self._replay_memory_dict: Dict[Any, FIFOReplayMemory] = {}

    def build(self) -> None:
        self._ops = self.get_ops(self._ops_name)
        state_dim = self._ops.policy_state_dim()
        action_dim = self._ops.policy_action_dim()
        self._replay_memory_dict = collections.defaultdict(lambda: FIFOReplayMemory(
            capacity=self._params.replay_memory_capacity,
            state_dim=state_dim,
            action_dim=action_dim
        ))

    def record(self, env_idx: int, exp_element: ExpElement) -> None:
        for agent_name in exp_element.agent_names:
            memory = self._replay_memory_dict[(env_idx, agent_name)]
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

    def get_local_ops_by_name(self, ops_name: str) -> AbsTrainOps:
        return DiscreteActorCriticOps(get_policy_func=self._get_policy_func, **self._params.extract_ops_params())

    async def train_step(self):
        batch_list = []
        for memory in self._replay_memory_dict.values():
            batch = self._ops.preprocess_batch(memory.sample(-1))  # Use all entries in the replay memory
            batch_list.append(batch)
        batch = merge_transition_batches(batch_list)

        for _ in range(self._params.grad_iters):
            batches = [batch] if self._params.data_parallelism == 1 else batch.split(self._params.data_parallelism)
            critic_grad_list = [self._ops.get_critic_grad(batch) for batch in batches]
            if isinstance(self._ops, RemoteOps):
                critic_grad_list = await asyncio.gather(*critic_grad_list)

            actor_grad_list = [self._ops.get_actor_grad(batch) for batch in batches]
            if isinstance(self._ops, RemoteOps):
                actor_grad_list = await asyncio.gather(*actor_grad_list)

            self._ops.update_critic(average_grads(critic_grad_list))
            self._ops.update_actor(average_grads(actor_grad_list))
