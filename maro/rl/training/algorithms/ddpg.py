# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# TODO: DDPG has net been tested in a real test case

import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

from maro.rl.model import QNet
from maro.rl.policy import ContinuousRLPolicy
from maro.rl.rollout import ExpElement
from maro.rl.training import AbsTrainOps, RandomReplayMemory, RemoteOps, SingleTrainer, TrainerParams, remote
from maro.rl.utils import TransitionBatch, average_grads, ndarray_to_tensor
from maro.utils import clone


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
    random_overwrite: bool = False

    def __post_init__(self) -> None:
        assert self.get_q_critic_net_func is not None

    def extract_ops_params(self) -> Dict[str, object]:
        return {
            "device": self.device,
            "get_q_critic_net_func": self.get_q_critic_net_func,
            "reward_discount": self.reward_discount,
            "q_value_loss_cls": self.q_value_loss_cls,
            "soft_update_coef": self.soft_update_coef
        }


class DDPGOps(AbsTrainOps):
    """Reference: https://spinningup.openai.com/en/latest/algorithms/ddpg.html"""
    def __init__(
        self,
        name: str,
        device: str,
        get_policy_func: Callable[[], ContinuousRLPolicy],
        get_q_critic_net_func: Callable[[], QNet],
        *,
        reward_discount: float,
        q_value_loss_cls: Callable = None,
        soft_update_coef: float = 1.0
    ) -> None:
        super(DDPGOps, self).__init__(
            name=name,
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
        self._soft_update_coef = soft_update_coef

    def _get_critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
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
        return self._q_value_loss_func(q_values, target_q_values)  # MSE(Q(s, a), y(r, s', d))

    @remote
    def get_critic_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        return self._q_critic_net.get_gradients(self._get_critic_loss(batch))

    def update_critic_with_grad(self, grad_dict: dict) -> None:
        self._q_critic_net.train()
        self._q_critic_net.apply_gradients(grad_dict)

    def update_critic(self, batch: TransitionBatch) -> None:
        self._q_critic_net.train()
        self._q_critic_net.step(self._get_critic_loss(batch))

    def _get_actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        assert self._is_valid_transition_batch(batch)
        self._policy.train()
        states = ndarray_to_tensor(batch.states, self._device)  # s

        policy_loss = -self._q_critic_net.q_values(
            states=states,  # s
            actions=self._policy.get_actions_tensor(states)  # miu(s)
        ).mean()  # -Q(s, miu(s))

        return policy_loss

    @remote
    def get_actor_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        return self._policy.get_gradients(self._get_actor_loss(batch))

    def update_actor_with_grad(self, grad_dict: dict) -> None:
        self._policy.train()
        self._policy.apply_gradients(grad_dict)

    def update_actor(self, batch: TransitionBatch) -> None:
        self._policy.train()
        self._policy.step(self._get_actor_loss(batch))

    def get_state(self) -> dict:
        return {
            "policy": self._policy.get_state(),
            "target_policy": self._target_policy.get_state(),
            "critic": self._q_critic_net.get_state(),
            "target_critic": self._target_q_critic_net.get_state()
        }

    def set_state(self, ops_state_dict: dict) -> None:
        self._policy.set_state(ops_state_dict["policy"])
        self._target_policy.set_state(ops_state_dict["target_policy"])
        self._q_critic_net.set_state(ops_state_dict["critic"])
        self._target_q_critic_net.set_state(ops_state_dict["target_critic"])

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

    def build(self) -> None:
        self._ops = self.get_ops(self._ops_name)
        self._replay_memory = RandomReplayMemory(
            capacity=self._params.replay_memory_capacity,
            state_dim=self._ops.policy_state_dim,
            action_dim=self._ops.policy_action_dim,
            random_overwrite=self._params.random_overwrite
        )

    def record(self, env_idx: int, exp_element: ExpElement) -> None:
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

    def get_local_ops_by_name(self, name: str) -> AbsTrainOps:
        return DDPGOps(name=name, get_policy_func=self._get_policy_func, **self._params.extract_ops_params())

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._batch_size)

    def train(self) -> None:
        assert not isinstance(self._ops, RemoteOps)
        for _ in range(self._params.num_epochs):
            batch = self._get_batch()
            self._ops.update_critic(batch)
            self._ops.update_actor(batch)

        self._policy_version += 1
        self._try_soft_update_target()

    async def train_as_task(self) -> None:
        assert isinstance(self._ops, RemoteOps)
        for _ in range(self._params.num_epochs):
            batch = self._get_batch()
            batches = [batch] if self._params.data_parallelism == 1 else batch.split(self._params.data_parallelism)
            # update critic
            critic_grad_list = await asyncio.gather(*[self._ops.get_critic_grad(batch) for batch in batches])
            self._ops.update_critic_with_grad(average_grads(critic_grad_list))
            # update actor
            actor_grad_list = await asyncio.gather(*[self._ops.get_actor_grad(batch) for batch in batches])
            self._ops.update_actor_with_grad(average_grads(actor_grad_list))

        self._policy_version += 1
        self._try_soft_update_target()

    def _try_soft_update_target(self) -> None:
        if self._policy_version - self._target_policy_version == self._params.update_target_every:
            self._ops.soft_update_target()
            self._target_policy_version = self._policy_version
