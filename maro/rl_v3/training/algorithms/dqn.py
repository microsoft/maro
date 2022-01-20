# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

from maro.rl_v3.policy import ValueBasedPolicy
from maro.rl_v3.rollout import ExpElement
from maro.rl_v3.training import AbsTrainOps, RandomReplayMemory, RemoteOps, SingleTrainer, TrainerParams, remote
from maro.rl_v3.utils import TransitionBatch, average_grads, ndarray_to_tensor
from maro.utils import clone


@dataclass
class DQNParams(TrainerParams):
    """
    reward_discount (float): Reward decay as defined in standard RL terminology. Defaults to 0.9.
    num_epochs (int): Number of training epochs per call to ``learn``. Defaults to 1.
    update_target_every (int): Number of gradient steps between target model updates. Defaults to 5.
    soft_update_coef (float): Soft update coefficient, e.g.,
        target_model = (soft_update_coef) * eval_model + (1-soft_update_coef) * target_model.
        Defaults to 0.1.
    double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
        i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
        See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
    random_overwrite (bool): This specifies overwrite behavior when the replay memory capacity is reached. If True,
        overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
        wrap-around. Defaults to False.
    """
    reward_discount: float = 0.9
    num_epochs: int = 1
    update_target_every: int = 5
    soft_update_coef: float = 0.1
    double: bool = False
    random_overwrite: bool = False
    data_parallelism: int = 1

    def extract_ops_params(self) -> Dict[str, object]:
        return {
            "device": self.device,
            "reward_discount": self.reward_discount,
            "soft_update_coef": self.soft_update_coef,
            "double": self.double,
        }


class DQNOps(AbsTrainOps):
    def __init__(
        self,
        device: str,
        get_policy_func: Callable[[], ValueBasedPolicy],
        *,
        reward_discount: float = 0.9,
        soft_update_coef: float = 0.1,
        double: bool = False
    ) -> None:
        super(DQNOps, self).__init__(
            device=device,
            is_single_scenario=True,
            get_policy_func=get_policy_func
        )

        assert isinstance(self._policy, ValueBasedPolicy)

        self._reward_discount = reward_discount
        self._soft_update_coef = soft_update_coef
        self._double = double
        self._loss_func = torch.nn.MSELoss()

        self._target_policy: ValueBasedPolicy = clone(self._policy)
        self._target_policy.set_name(f"target_{self._policy.name}")
        self._target_policy.eval()
        self._target_policy.to_device(self._device)

    @remote
    def get_batch_grad(self, batch: TransitionBatch) -> Dict[str, Dict[str, torch.Tensor]]:
        assert self._is_valid_transition_batch(batch)
        self._policy.train()
        states = ndarray_to_tensor(batch.states, self._device)
        next_states = ndarray_to_tensor(batch.next_states, self._device)
        actions = ndarray_to_tensor(batch.actions, self._device)
        rewards = ndarray_to_tensor(batch.rewards, self._device)
        terminals = ndarray_to_tensor(batch.terminals, self._device).float()

        with torch.no_grad():
            if self._double:
                self._policy.exploit()
                actions_by_eval_policy = self._policy.get_actions_tensor(next_states)
                next_q_values = self._target_policy.q_values_tensor(next_states, actions_by_eval_policy)
            else:
                self._target_policy.exploit()
                actions = self._target_policy.get_actions_tensor(next_states)
                next_q_values = self._target_policy.q_values_tensor(next_states, actions)
        target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()

        q_values = self._policy.q_values_tensor(states, actions)
        loss: torch.Tensor = self._loss_func(q_values, target_q_values)

        return self._policy.get_gradients(loss)

    def get_state(self, scope: str = "all") -> dict:
        return {
            "policy_state": self._policy.get_state(),
            "target_q_net_state": self._target_policy.get_state()
        }

    def set_state(self, ops_state_dict: dict, scope: str = "all") -> None:
        self._policy.set_state(ops_state_dict["policy_state"])
        self._target_policy.set_state(ops_state_dict["target_q_net_state"])

    def update(self, grad_dict: dict) -> None:
        self._policy.train()
        self._policy.apply_gradients(grad_dict)

    def soft_update_target(self) -> None:
        self._target_policy.soft_update(self._policy, self._soft_update_coef)


class DQN(SingleTrainer):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.
    """
    def __init__(self, name: str, params: DQNParams) -> None:
        super(DQN, self).__init__(name, params)
        self._params = params
        self._q_net_version = self._target_q_net_version = 0
        self._ops_name = f"{self._name}.ops"
        self._ops = self.get_ops(self._ops_name)
        self._replay_memory = RandomReplayMemory(
            capacity=self._params.replay_memory_capacity,
            state_dim=self._ops.policy_state_dim(),
            action_dim=self._ops.policy_action_dim(),
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

    def get_local_ops_by_name(self, ops_name: str) -> AbsTrainOps:
        return DQNOps(get_policy_func=self._get_policy_func, **self._params.extract_ops_params())

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._batch_size)

    async def train_step(self) -> None:
        for _ in range(self._params.num_epochs):
            batch = self._get_batch()
            batches = [batch] if self._params.data_parallelism == 1 else batch.split(self._params.data_parallelism)
            grad_list = [self._ops.get_batch_grad(batch) for batch in batches]
            if isinstance(self._ops, RemoteOps):
                grad_list = await asyncio.gather(*grad_list)
            self._ops.update(average_grads(grad_list))

        self._q_net_version += 1
        if self._q_net_version - self._target_q_net_version == self._params.update_target_every:
            self._ops.soft_update_target()
            self._target_q_net_version = self._q_net_version
