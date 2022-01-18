# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# TODO: DDPG has net been tested in a real test case

import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

from maro.rl_v3.model import QNet
from maro.rl_v3.policy import ContinuousRLPolicy
from maro.rl_v3.rollout import ExpElement
from maro.rl_v3.training import AbsTrainOps, RandomReplayMemory, SingleTrainer, TrainerParams
from maro.rl_v3.utils import TransitionBatch, average_grads, ndarray_to_tensor
from maro.utils import clone

from ..train_ops import RemoteOps, remote


@dataclass
class DDPGParams(TrainerParams):
    """
    get_q_critic_net_func (Callable[[], QNet]): Function to get Q critic net.
    reward_discount (float): Reward decay as defined in standard RL terminology.
    num_epochs (int): Number of training epochs per call to ``learn``. Defaults to 1.
    update_target_every (int): Number of training rounds between policy target model updates.
    q_value_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for
        the Q-value loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
    soft_update_coef (float): Soft update coefficient, e.g., target_model = (soft_update_coef) * eval_model +
        (1-soft_update_coef) * target_model. Defaults to 1.0.
    critic_loss_coef (float): Coefficient for critic loss in total loss. Defaults to 0.1.
    random_overwrite (bool): This specifies overwrite behavior when the replay memory capacity is reached. If True,
        overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
        wrap-around. Defaults to False.
    """
    get_q_critic_net_func: Callable[[], QNet] = None
    reward_discount: float = 0.9
    num_epochs: int = 1
    update_target_every: int = 5
    q_value_loss_cls: Callable = None
    soft_update_coef: float = 1.0
    critic_loss_coef: float = 0.1
    random_overwrite: bool = False
    data_parallelism: int = 1

    def __post_init__(self) -> None:
        assert self.get_q_critic_net_func is not None

    def extract_ops_params(self) -> Dict[str, object]:
        return {
            "device": self.device,
            "get_q_critic_net_func": self.get_q_critic_net_func,
            "reward_discount": self.reward_discount,
            "q_value_loss_cls": self.q_value_loss_cls,
            "soft_update_coef": self.soft_update_coef,
            "critic_loss_coef": self.critic_loss_coef,
        }


class DDPGOps(AbsTrainOps):
    """Reference: https://spinningup.openai.com/en/latest/algorithms/ddpg.html"""
    def __init__(
        self,
        device: str,
        get_policy_func: Callable[[], ContinuousRLPolicy],
        get_q_critic_net_func: Callable[[], QNet],
        *,
        reward_discount: float,
        q_value_loss_cls: Callable = None,
        soft_update_coef: float = 1.0,
        critic_loss_coef: float = 0.1
    ) -> None:
        super(DDPGOps, self).__init__(
            device=device,
            is_single_scenario=True,
            get_policy_func=get_policy_func
        )

        assert isinstance(self._policy, ContinuousRLPolicy)

        self._target_policy = clone(self._policy)
        self._target_policy.set_name(f"target_{self._policy.name}")
        self._target_policy.eval()
        self._target_policy.to_device(self._device)
        self._q_critic_net = get_q_critic_net_func()
        self._q_critic_net.to(self._device)
        self._target_q_critic_net: QNet = clone(self._q_critic_net)
        self._target_q_critic_net.eval()
        self._target_q_critic_net.to(self._device)

        self._reward_discount = reward_discount
        self._q_value_loss_func = q_value_loss_cls() if q_value_loss_cls is not None else torch.nn.MSELoss()
        self._critic_loss_coef = critic_loss_coef
        self._soft_update_coef = soft_update_coef

    @remote
    def get_critic_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        assert self._is_valid_transition_batch(batch)
        self._q_critic_net.train()
        states = ndarray_to_tensor(batch.states, self._device)  # s
        next_states = ndarray_to_tensor(batch.next_states, self._device)  # s'
        actions = ndarray_to_tensor(batch.actions, self._device)  # a
        rewards = ndarray_to_tensor(batch.rewards, self._device)  # r
        terminals = ndarray_to_tensor(batch.terminals, self._device)  # d

        with torch.no_grad():
            next_q_values = self._target_q_critic_net.q_values(
                states=next_states,  # s'
                actions=self._target_policy.get_actions_tensor(next_states)  # miu_targ(s')
            )  # Q_targ(s', miu_targ(s'))

        # y(r, s', d) = r + gamma * (1 - d) * Q_targ(s', miu_targ(s'))
        target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()

        q_values = self._q_critic_net.q_values(states=states, actions=actions)  # Q(s, a)
        critic_loss = self._q_value_loss_func(q_values, target_q_values)  # MSE(Q(s, a), y(r, s', d))

        return self._q_critic_net.get_gradients(critic_loss * self._critic_loss_coef)

    @remote
    def get_actor_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        assert self._is_valid_transition_batch(batch)
        self._policy.train()
        states = ndarray_to_tensor(batch.states, self._device)  # s

        policy_loss = -self._q_critic_net.q_values(
            states=states,  # s
            actions=self._policy.get_actions_tensor(states)  # miu(s)
        ).mean()  # -Q(s, miu(s))

        return self._policy.get_gradients(policy_loss)

    def update_critic(self, grad_dict: dict) -> None:
        self._q_critic_net.train()
        self._q_critic_net.apply_gradients(grad_dict)

    def update_actor(self, grad_dict: dict) -> None:
        self._policy.train()
        self._policy.apply_gradients(grad_dict)

    def get_state(self, scope: str = "all") -> dict:
        ret_dict = {}
        if scope in ("all", "actor"):
            ret_dict["policy_state"] = self._policy.get_state()
            ret_dict["target_policy_state"] = self._target_policy.get_state()
        if scope in ("all", "critic"):
            ret_dict["critic_state"] = self._q_critic_net.get_net_state()
            ret_dict["target_critic_state"] = self._target_q_critic_net.get_net_state()
        return ret_dict

    def set_state(self, ops_state_dict: dict, scope: str = "all") -> None:
        if scope in ("all", "actor"):
            self._policy.set_state(ops_state_dict["policy_state"])
            self._target_policy.set_state(ops_state_dict["target_policy_state"])
        if scope in ("all", "critic"):
            self._q_critic_net.set_net_state(ops_state_dict["critic_state"])
            self._target_q_critic_net.set_net_state(ops_state_dict["target_critic_state"])

    def soft_update_target(self) -> None:
        self._target_policy.soft_update(self._policy, self._soft_update_coef)
        self._target_q_critic_net.soft_update(self._q_critic_net, self._soft_update_coef)


class DDPG(SingleTrainer):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        https://arxiv.org/pdf/1509.02971.pdf
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg
    """

    def __init__(self, name: str, params: DDPGParams) -> None:
        super(DDPG, self).__init__(name, params)
        self._params = params
        self._policy_version = self._target_policy_version = 0
        self._ops_name = f"{self._name}.ops"

        self._replay_memory: Optional[RandomReplayMemory] = None

    async def build(self) -> None:
        self._ops = self.get_ops(self._ops_name)
        state_dim = await self._ops.policy_state_dim()
        action_dim = await self._ops.policy_action_dim()
        self._replay_memory = RandomReplayMemory(
            capacity=self._params.replay_memory_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            random_overwrite=self._params.random_overwrite
        )

    def record(self, exp_element: ExpElement) -> None:
        for agent_name in exp_element.agent_names:
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
            self._replay_memory.put(transition_batch)

    def get_local_ops_by_name(self, ops_name: str) -> AbsTrainOps:
        return DDPGOps(get_policy_func=self._get_policy_func, **self._params.extract_ops_params())

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._batch_size)

    async def train_step(self) -> None:
        for _ in range(self._params.num_epochs):
            batch = self._get_batch()
            batches = [batch] if self._params.data_parallelism == 1 else batch.split(self._params.data_parallelism)
            # update critic
            critic_grad_list = [self._ops.get_critic_grad(batch) for batch in batches]
            if isinstance(self._ops, RemoteOps):
                critic_grad_list = await asyncio.gather(*critic_grad_list)
            self._ops.update_critic(average_grads(critic_grad_list))
            # update actor
            actor_grad_list = [self._ops.get_actor_grad(batch) for batch in batches]
            if isinstance(self._ops, RemoteOps):
                actor_grad_list = await asyncio.gather(*actor_grad_list)
            self._ops.update_actor(average_grads(actor_grad_list))

        self._policy_version += 1
        if self._policy_version - self._target_policy_version == self._params.update_target_every:
            self._ops.soft_update_target()
            self._target_policy_version = self._policy_version
