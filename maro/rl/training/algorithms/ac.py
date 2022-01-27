# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import collections
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from maro.rl.model import VNet
from maro.rl.policy import DiscretePolicyGradient
from maro.rl.rollout import ExpElement
from maro.rl.training import AbsTrainOps, FIFOReplayMemory, RemoteOps, SingleTrainer, TrainerParams, remote
from maro.rl.utils import TransitionBatch, average_grads, discount_cumsum, merge_transition_batches, ndarray_to_tensor


@dataclass
class DiscreteActorCriticParams(TrainerParams):
    """
    get_v_critic_net_func (Callable[[], VNet]): Function to get V critic net.
    reward_discount (float): Reward decay as defined in standard RL terminology. Defaults to 0.9.
    grad_iters (int): Number of iterations to calculate gradients. Defaults to 1.
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
    critic_loss_cls: Callable = None
    clip_ratio: float = None
    lam: float = 0.9
    min_logp: Optional[float] = None

    def __post_init__(self) -> None:
        assert self.get_v_critic_net_func is not None

    def extract_ops_params(self) -> Dict[str, object]:
        return {
            "device": self.device,
            "get_v_critic_net_func": self.get_v_critic_net_func,
            "reward_discount": self.reward_discount,
            "critic_loss_cls": self.critic_loss_cls,
            "clip_ratio": self.clip_ratio,
            "lam": self.lam,
            "min_logp": self.min_logp,
        }


class DiscreteActorCriticOps(AbsTrainOps):
    def __init__(
        self,
        name: str,
        device: str,
        get_policy_func: Callable[[], DiscretePolicyGradient],
        get_v_critic_net_func: Callable[[], VNet],
        *,
        reward_discount: float = 0.9,
        critic_loss_cls: Callable = None,
        clip_ratio: float = None,
        lam: float = 0.9,
        min_logp: float = None
    ) -> None:
        super(DiscreteActorCriticOps, self).__init__(
            name=name,
            device=device,
            is_single_scenario=True,
            get_policy_func=get_policy_func
        )

        assert isinstance(self._policy, DiscretePolicyGradient)

        self._reward_discount = reward_discount
        self._critic_loss_func = critic_loss_cls() if critic_loss_cls is not None else torch.nn.MSELoss()
        self._clip_ratio = clip_ratio
        self._lam = lam
        self._min_logp = min_logp
        self._v_critic_net = get_v_critic_net_func()
        self._v_critic_net.to(self._device)

    def _get_critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        self._v_critic_net.train()
        states = ndarray_to_tensor(batch.states, self._device)  # s
        state_values = self._v_critic_net.v_values(states)
        returns = ndarray_to_tensor(batch.returns, self._device)
        return self._critic_loss_func(state_values, returns)

    @remote
    def get_critic_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        return self._v_critic_net.get_gradients(self._get_critic_loss(batch))

    def update_critic(self, batch: TransitionBatch) -> None:
        self._v_critic_net.step(self._get_critic_loss(batch))

    def update_critic_with_grad(self, grad_dict: dict) -> None:
        """Reference: https://tinyurl.com/2ezte4cr
        """
        self._v_critic_net.train()
        self._v_critic_net.apply_gradients(grad_dict)

    def _get_actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
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

        return actor_loss

    @remote
    def get_actor_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        return self._policy.get_gradients(self._get_actor_loss(batch))

    def update_actor(self, batch: TransitionBatch) -> None:
        self._policy.step(self._get_actor_loss(batch))

    def update_actor_with_grad(self, grad_dict: dict) -> None:
        """Reference: https://tinyurl.com/2ezte4cr
        """
        self._policy.train()
        self._policy.apply_gradients(grad_dict)

    def get_state(self) -> dict:
        return {
            "policy": self._policy.get_state(),
            "critic": self._v_critic_net.get_state()
        }

    def set_state(self, ops_state_dict: dict) -> None:
        self._policy.set_state(ops_state_dict["policy"])
        self._v_critic_net.set_state(ops_state_dict["critic"])

    def _preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
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

    def preprocess_and_merge_batches(self, batch_list: List[TransitionBatch]) -> TransitionBatch:
        return merge_transition_batches([self._preprocess_batch(batch) for batch in batch_list])


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
        self._replay_memory_dict = collections.defaultdict(lambda: FIFOReplayMemory(
            capacity=self._params.replay_memory_capacity,
            state_dim=self._ops.policy_state_dim,
            action_dim=self._ops.policy_action_dim
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

    def get_local_ops_by_name(self, name: str) -> AbsTrainOps:
        return DiscreteActorCriticOps(
            name=name, get_policy_func=self._get_policy_func, **self._params.extract_ops_params()
        )

    def _get_batch(self) -> TransitionBatch:
        batch_list = [memory.sample(-1) for memory in self._replay_memory_dict.values()]
        return self._ops.preprocess_and_merge_batches(batch_list)

    def train(self):
        assert not isinstance(self._ops, RemoteOps)
        batch = self._get_batch()
        for _ in range(self._params.grad_iters):
            self._ops.update_critic(batch)
            self._ops.update_actor(batch)

    async def train_as_task(self):
        assert isinstance(self._ops, RemoteOps)
        batch = self._get_batch()
        for _ in range(self._params.grad_iters):
            batches = [batch] if self._params.data_parallelism == 1 else batch.split(self._params.data_parallelism)
            critic_grad_list = await asyncio.gather(*[self._ops.get_critic_grad(batch) for batch in batches])
            actor_grad_list = await asyncio.gather(*[self._ops.get_actor_grad(batch) for batch in batches])
            self._ops.update_critic_with_grad(average_grads(critic_grad_list))
            self._ops.update_actor_with_grad(average_grads(actor_grad_list))
