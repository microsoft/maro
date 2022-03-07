# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# TODO: DDPG has net been tested in a real test case

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

from maro.rl.model import QNet
from maro.rl.policy import ContinuousRLPolicy
from maro.rl.rollout import ExpElement
from maro.rl.training import AbsTrainOps, RandomReplayMemory, RemoteOps, SingleAlgorithm, AlgorithmParams, remote
from maro.rl.utils import TransitionBatch, ndarray_to_tensor
from maro.utils import clone


@dataclass
class DDPGParams(AlgorithmParams):
    """
    get_q_critic_net_func (Callable[[], QNet]): Function to get Q critic net.
    reward_discount (float, default=0.9): Reward decay as defined in standard RL terminology.
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
            "soft_update_coef": self.soft_update_coef,
            "data_parallelism": self.data_parallelism,
        }


class DDPGOps(AbsTrainOps):
    """DDPG algorithm implementation. Reference: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    """

    def __init__(
        self,
        name: str,
        device: str,
        get_policy_func: Callable[[], ContinuousRLPolicy],
        get_q_critic_net_func: Callable[[], QNet],
        parallelism: int = 1,
        *,
        reward_discount: float,
        q_value_loss_cls: Callable = None,
        soft_update_coef: float = 1.0,
    ) -> None:
        super(DDPGOps, self).__init__(
            name=name,
            device=device,
            is_single_scenario=True,
            get_policy_func=get_policy_func,
            parallelism=parallelism,
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
        """Compute the critic loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The critic loss of the batch.
        """
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
                actions=self._target_policy.get_actions_tensor(next_states),  # miu_targ(s')
            )  # Q_targ(s', miu_targ(s'))

        # y(r, s', d) = r + gamma * (1 - d) * Q_targ(s', miu_targ(s'))
        target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()
        q_values = self._q_critic_net.q_values(states=states, actions=actions)  # Q(s, a)
        return self._q_value_loss_func(q_values, target_q_values)  # MSE(Q(s, a), y(r, s', d))

    @remote
    def get_critic_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        """Compute the critic network's gradients of a batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            grad (torch.Tensor): The critic gradient of the batch.
        """
        return self._q_critic_net.get_gradients(self._get_critic_loss(batch))

    def update_critic_with_grad(self, grad_dict: dict) -> None:
        """Update the critic network with remotely computed gradients.

        Args:
            grad_dict (dict): Gradients.
        """
        self._q_critic_net.train()
        self._q_critic_net.apply_gradients(grad_dict)

    def update_critic(self, batch: TransitionBatch) -> None:
        """Update the critic network using a batch.

        Args:
            batch (TransitionBatch): Batch.
        """
        self._q_critic_net.train()
        self._q_critic_net.train_step(self._get_critic_loss(batch))

    def _get_actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """Compute the actor loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The actor loss of the batch.
        """
        assert self._is_valid_transition_batch(batch)
        self._policy.train_step()
        states = ndarray_to_tensor(batch.states, self._device)  # s

        policy_loss = -self._q_critic_net.q_values(
            states=states,  # s
            actions=self._policy.get_actions_tensor(states),  # miu(s)
        ).mean()  # -Q(s, miu(s))

        return policy_loss

    @remote
    def get_actor_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        """Compute the actor network's gradients of a batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            grad (torch.Tensor): The actor gradient of the batch.
        """
        return self._policy.get_gradients(self._get_actor_loss(batch))

    def update_actor_with_grad(self, grad_dict: dict) -> None:
        """Update the actor network with remotely computed gradients.

        Args:
            grad_dict (dict): Gradients.
        """
        self._policy.train_step()
        self._policy.apply_gradients(grad_dict)

    def update_actor(self, batch: TransitionBatch) -> None:
        """Update the actor network using a batch.

        Args:
            batch (TransitionBatch): Batch.
        """
        self._policy.train_step()
        self._policy.train_step(self._get_actor_loss(batch))

    def get_state(self) -> dict:
        return {
            "policy": self._policy.get_state(),
            "target_policy": self._target_policy.get_state(),
            "critic": self._q_critic_net.get_state(),
            "target_critic": self._target_q_critic_net.get_state(),
        }

    def set_state(self, ops_state_dict: dict) -> None:
        self._policy.set_state(ops_state_dict["policy"])
        self._target_policy.set_state(ops_state_dict["target_policy"])
        self._q_critic_net.set_state(ops_state_dict["critic"])
        self._target_q_critic_net.set_state(ops_state_dict["target_critic"])

    def soft_update_target(self) -> None:
        """Soft update the target policy and target critic.
        """
        self._target_policy.soft_update(self._policy, self._soft_update_coef)
        self._target_q_critic_net.soft_update(self._q_critic_net, self._soft_update_coef)


class DDPG(SingleAlgorithm):
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
            random_overwrite=self._params.random_overwrite,
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
        return DDPGOps(
            name=name, get_policy_func=self._get_policy_func, parallelism=self._params.data_parallelism,
            **self._params.extract_ops_params(),
        )

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._batch_size)

    def train_step(self) -> None:
        assert isinstance(self._ops, DDPGOps)
        for _ in range(self._params.num_epochs):
            batch = self._get_batch()
            self._ops.update_critic(batch)
            self._ops.update_actor(batch)

        self._try_soft_update_target()

    async def train_step_as_task(self) -> None:
        assert isinstance(self._ops, RemoteOps)
        for _ in range(self._params.num_epochs):
            batch = self._get_batch()
            self._ops.update_critic_with_grad(await self._ops.get_critic_grad(batch))
            self._ops.update_actor_with_grad(await self._ops.get_actor_grad(batch))

        self._try_soft_update_target()

    def _try_soft_update_target(self) -> None:
        """Soft update the target policy and target critic.
        """
        self._policy_version += 1
        if self._policy_version - self._target_policy_version == self._params.update_target_every:
            self._ops.soft_update_target()
            self._target_policy_version = self._policy_version
