# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch

from maro.rl.model import MultiQNet
from maro.rl.policy import DiscretePolicyGradient, RLPolicy
from maro.rl.rollout import ExpElement
from maro.rl.training import (
    AbsTrainOps,
    BaseTrainerParams,
    MultiAgentTrainer,
    RandomMultiReplayMemory,
    RemoteOps,
    remote,
)
from maro.rl.utils import MultiTransitionBatch, get_torch_device, ndarray_to_tensor
from maro.rl.utils.objects import FILE_SUFFIX
from maro.utils import clone


@dataclass
class DiscreteMADDPGParams(BaseTrainerParams):
    """
    get_q_critic_net_func (Callable[[], MultiQNet]): Function to get multi Q critic net.
    num_epochs (int, default=10): Number of training epochs.
    update_target_every (int, default=5): Number of gradient steps between target model updates.
    soft_update_coef (float, default=0.5): Soft update coefficient, e.g.,
        target_model = (soft_update_coef) * eval_model + (1-soft_update_coef) * target_model.
    q_value_loss_cls (Callable, default=None): Critic loss function. If it is None, use MSE.
    shared_critic (bool, default=False): Whether different policies use shared critic or individual policies.
    """

    get_q_critic_net_func: Callable[[], MultiQNet]
    num_epoch: int = 10
    update_target_every: int = 5
    soft_update_coef: float = 0.5
    q_value_loss_cls: Optional[Callable] = None
    shared_critic: bool = False


class DiscreteMADDPGOps(AbsTrainOps):
    def __init__(
        self,
        name: str,
        policy: RLPolicy,
        param: DiscreteMADDPGParams,
        shared_critic: bool,
        policy_idx: int,
        parallelism: int = 1,
        reward_discount: float = 0.9,
    ) -> None:
        super(DiscreteMADDPGOps, self).__init__(
            name=name,
            policy=policy,
            parallelism=parallelism,
        )

        self._policy_idx = policy_idx
        self._shared_critic = shared_critic

        # Actor
        if self._policy:
            assert isinstance(self._policy, DiscretePolicyGradient)
            self._target_policy: DiscretePolicyGradient = clone(self._policy)
            self._target_policy.set_name(f"target_{self._policy.name}")
            self._target_policy.eval()

        # Critic
        self._q_critic_net: MultiQNet = param.get_q_critic_net_func()
        self._target_q_critic_net: MultiQNet = clone(self._q_critic_net)
        self._target_q_critic_net.eval()

        self._reward_discount = reward_discount
        self._q_value_loss_func = param.q_value_loss_cls() if param.q_value_loss_cls is not None else torch.nn.MSELoss()
        self._update_target_every = param.update_target_every
        self._soft_update_coef = param.soft_update_coef

    def get_target_action(self, batch: MultiTransitionBatch) -> torch.Tensor:
        """Get the target policies' actions according to the batch.

        Args:
            batch (MultiTransitionBatch): Batch.

        Returns:
            actions (torch.Tensor): Target policies' actions.
        """
        agent_state = ndarray_to_tensor(batch.agent_states[self._policy_idx], device=self._device)
        return self._target_policy.get_actions_tensor(agent_state)

    def get_latest_action(self, batch: MultiTransitionBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the latest actions and corresponding log-probabilities according to the batch.

        Args:
            batch (MultiTransitionBatch): Batch.

        Returns:
            actions (torch.Tensor): Target policies' actions.
            logps (torch.Tensor): Log-probabilities.
        """
        assert isinstance(self._policy, DiscretePolicyGradient)

        agent_state = ndarray_to_tensor(batch.agent_states[self._policy_idx], device=self._device)
        self._policy.train()
        action = self._policy.get_actions_tensor(agent_state)
        logps = self._policy.get_states_actions_logps(agent_state, action)
        return action, logps

    def _get_critic_loss(self, batch: MultiTransitionBatch, next_actions: List[torch.Tensor]) -> torch.Tensor:
        """Compute the critic loss of the batch.

        Args:
            batch (MultiTransitionBatch): Batch.
            next_actions (List[torch.Tensor]): List of next actions of all policies.

        Returns:
            loss (torch.Tensor): The critic loss of the batch.
        """
        assert not self._shared_critic
        assert isinstance(next_actions, list) and all(isinstance(action, torch.Tensor) for action in next_actions)

        states = ndarray_to_tensor(batch.states, device=self._device)  # x
        actions = [ndarray_to_tensor(action, device=self._device) for action in batch.actions]  # a
        next_states = ndarray_to_tensor(batch.next_states, device=self._device)  # x'
        rewards = ndarray_to_tensor(np.vstack([reward for reward in batch.rewards]), device=self._device)  # r
        terminals = ndarray_to_tensor(batch.terminals, device=self._device)  # d

        self._q_critic_net.train()
        with torch.no_grad():
            next_q_values = self._target_q_critic_net.q_values(
                states=next_states,  # x'
                actions=next_actions,
            )  # a'
        target_q_values = rewards[self._policy_idx] + self._reward_discount * (1 - terminals.float()) * next_q_values
        q_values = self._q_critic_net.q_values(
            states=states,  # x
            actions=actions,  # a
        )  # Q(x, a)
        return self._q_value_loss_func(q_values, target_q_values.detach())

    @remote
    def get_critic_grad(
        self,
        batch: MultiTransitionBatch,
        next_actions: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the critic network's gradients of a batch.

        Args:
            batch (MultiTransitionBatch): Batch.
            next_actions (List[torch.Tensor]): List of next actions of all policies.

        Returns:
            grad (torch.Tensor): The critic gradient of the batch.
        """
        return self._q_critic_net.get_gradients(self._get_critic_loss(batch, next_actions))

    def update_critic(self, batch: MultiTransitionBatch, next_actions: List[torch.Tensor]) -> None:
        """Update the critic network using a batch.

        Args:
            batch (MultiTransitionBatch): Batch.
            next_actions (List[torch.Tensor]): List of next actions of all policies.
        """
        self._q_critic_net.train()
        self._q_critic_net.step(self._get_critic_loss(batch, next_actions))

    def update_critic_with_grad(self, grad_dict: dict) -> None:
        """Update the critic network with remotely computed gradients.

        Args:
            grad_dict (dict): Gradients.
        """
        self._q_critic_net.train()
        self._q_critic_net.apply_gradients(grad_dict)

    def _get_actor_loss(self, batch: MultiTransitionBatch) -> torch.Tensor:
        """Compute the actor loss of the batch.

        Args:
            batch (MultiTransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The actor loss of the batch.
        """
        latest_action, latest_action_logp = self.get_latest_action(batch)
        states = ndarray_to_tensor(batch.states, device=self._device)  # x
        actions = [ndarray_to_tensor(action, device=self._device) for action in batch.actions]  # a
        actions[self._policy_idx] = latest_action
        self._policy.train()
        self._q_critic_net.freeze()
        actor_loss = -(
            self._q_critic_net.q_values(
                states=states,  # x
                actions=actions,  # [a^j_1, ..., a_i, ..., a^j_N]
            )
            * latest_action_logp
        ).mean()  # Q(x, a^j_1, ..., a_i, ..., a^j_N)
        self._q_critic_net.unfreeze()
        return actor_loss

    @remote
    def get_actor_grad(self, batch: MultiTransitionBatch) -> Dict[str, torch.Tensor]:
        """Compute the actor network's gradients of a batch.

        Args:
            batch (MultiTransitionBatch): Batch.

        Returns:
            grad (torch.Tensor): The actor gradient of the batch.
        """
        return self._policy.get_gradients(self._get_actor_loss(batch))

    def update_actor(self, batch: MultiTransitionBatch) -> None:
        """Update the actor network using a batch.

        Args:
            batch (MultiTransitionBatch): Batch.
        """
        self._policy.train()
        self._policy.train_step(self._get_actor_loss(batch))

    def update_actor_with_grad(self, grad_dict: dict) -> None:
        """Update the critic network with remotely computed gradients.

        Args:
            grad_dict (dict): Gradients.
        """
        self._policy.train()
        self._policy.apply_gradients(grad_dict)

    def soft_update_target(self) -> None:
        """Soft update the target policies and target critics."""
        if self._policy:
            self._target_policy.soft_update(self._policy, self._soft_update_coef)
        if not self._shared_critic:
            self._target_q_critic_net.soft_update(self._q_critic_net, self._soft_update_coef)

    def get_critic_state(self) -> dict:
        return {
            "critic": self._q_critic_net.get_state(),
            "target_critic": self._target_q_critic_net.get_state(),
        }

    def set_critic_state(self, ops_state_dict: dict) -> None:
        self._q_critic_net.set_state(ops_state_dict["critic"])
        self._target_q_critic_net.set_state(ops_state_dict["target_critic"])

    def get_actor_state(self) -> dict:
        if self._policy:
            return {"policy": self._policy.get_state(), "target_policy": self._target_policy.get_state()}
        else:
            return {}

    def set_actor_state(self, ops_state_dict: dict) -> None:
        if self._policy:
            self._policy.set_state(ops_state_dict["policy"])
            self._target_policy.set_state(ops_state_dict["target_policy"])

    def get_non_policy_state(self) -> dict:
        return self.get_critic_state()

    def set_non_policy_state(self, state: dict) -> None:
        self.set_critic_state(state)

    def to_device(self, device: str = None) -> None:
        self._device = get_torch_device(device)
        if self._policy:
            self._policy.to_device(self._device)
            self._target_policy.to_device(self._device)

        self._q_critic_net.to(self._device)
        self._target_q_critic_net.to(self._device)


class DiscreteMADDPGTrainer(MultiAgentTrainer):
    """Multi-agent deep deterministic policy gradient (MADDPG) algorithm adapted for discrete action space.

    See https://arxiv.org/abs/1706.02275 for details.
    """

    def __init__(
        self,
        name: str,
        params: DiscreteMADDPGParams,
        replay_memory_capacity: int = 10000,
        batch_size: int = 128,
        data_parallelism: int = 1,
        reward_discount: float = 0.9,
    ) -> None:
        super(DiscreteMADDPGTrainer, self).__init__(
            name,
            replay_memory_capacity,
            batch_size,
            data_parallelism,
            reward_discount,
        )
        self._params = params

        self._state_dim = params.get_q_critic_net_func().state_dim
        self._policy_version = self._target_policy_version = 0
        self._shared_critic_ops_name = f"{self._name}.shared_critic"

        self._actor_ops_list: List[DiscreteMADDPGOps] = []
        self._critic_ops: Optional[DiscreteMADDPGOps] = None
        self._policy2agent: Dict[str, str] = {}
        self._ops_dict: Dict[str, DiscreteMADDPGOps] = {}

    def build(self) -> None:
        self._placeholder_policy = self._policy_dict[self._policy_names[0]]

        for policy in self._policy_dict.values():
            self._ops_dict[policy.name] = cast(DiscreteMADDPGOps, self.get_ops(policy.name))

        self._actor_ops_list = list(self._ops_dict.values())

        if self._params.shared_critic:
            assert self._critic_ops is not None
            self._ops_dict[self._shared_critic_ops_name] = cast(
                DiscreteMADDPGOps,
                self.get_ops(self._shared_critic_ops_name),
            )
            self._critic_ops = self._ops_dict[self._shared_critic_ops_name]

        self._replay_memory = RandomMultiReplayMemory(
            capacity=self._replay_memory_capacity,
            state_dim=self._state_dim,
            action_dims=[ops.policy_action_dim for ops in self._actor_ops_list],
            agent_states_dims=[ops.policy_state_dim for ops in self._actor_ops_list],
        )

        assert len(self._agent2policy.keys()) == len(self._agent2policy.values())  # agent <=> policy
        self._policy2agent = {policy_name: agent_name for agent_name, policy_name in self._agent2policy.items()}

    def record_multiple(self, env_idx: int, exp_elements: List[ExpElement]) -> None:
        terminal_flags: List[bool] = []
        for exp_element in exp_elements:
            assert exp_element.num_agents == len(self._agent2policy.keys())

            if min(exp_element.terminal_dict.values()) != max(exp_element.terminal_dict.values()):
                raise ValueError("The 'terminal` flag of all agents at every tick must be identical.")
            terminal_flags.append(min(exp_element.terminal_dict.values()))

        actions: List[np.ndarray] = []
        rewards: List[np.ndarray] = []
        agent_states: List[np.ndarray] = []
        next_agent_states: List[np.ndarray] = []
        for policy_name in self._policy_dict:
            agent_name = self._policy2agent[policy_name]
            actions.append(np.vstack([exp_element.action_dict[agent_name] for exp_element in exp_elements]))
            rewards.append(np.array([exp_element.reward_dict[agent_name] for exp_element in exp_elements]))
            agent_states.append(np.vstack([exp_element.agent_state_dict[agent_name] for exp_element in exp_elements]))
            next_agent_states.append(
                np.vstack(
                    [
                        exp_element.next_agent_state_dict.get(agent_name, exp_element.agent_state_dict[agent_name])
                        for exp_element in exp_elements
                    ],
                ),
            )

        transition_batch = MultiTransitionBatch(
            states=np.vstack([exp_element.state for exp_element in exp_elements]),
            actions=actions,
            rewards=rewards,
            next_states=np.vstack(
                [
                    exp_element.next_state if exp_element.next_state is not None else exp_element.state
                    for exp_element in exp_elements
                ],
            ),
            agent_states=agent_states,
            next_agent_states=next_agent_states,
            terminals=np.array(terminal_flags),
        )
        self._replay_memory.put(transition_batch)

    def get_local_ops(self, name: str) -> AbsTrainOps:
        if name == self._shared_critic_ops_name:
            return DiscreteMADDPGOps(
                name=name,
                policy=self._placeholder_policy,
                param=self._params,
                shared_critic=False,
                policy_idx=-1,
                parallelism=self._data_parallelism,
                reward_discount=self._reward_discount,
            )
        else:
            return DiscreteMADDPGOps(
                name=name,
                policy=self._policy_dict[name],
                param=self._params,
                shared_critic=self._params.shared_critic,
                policy_idx=self._policy_names.index(name),
                parallelism=self._data_parallelism,
                reward_discount=self._reward_discount,
            )

    def _get_batch(self, batch_size: int = None) -> MultiTransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._batch_size)

    def train_step(self) -> None:
        assert not self._params.shared_critic or isinstance(self._critic_ops, DiscreteMADDPGOps)
        assert all(isinstance(ops, DiscreteMADDPGOps) for ops in self._actor_ops_list)
        for _ in range(self._params.num_epoch):
            batch = self._get_batch()
            # Collect next actions
            next_actions = [ops.get_target_action(batch) for ops in self._actor_ops_list]

            # Update critic
            if self._params.shared_critic:
                assert self._critic_ops is not None
                self._critic_ops.update_critic(batch, next_actions)
                critic_state_dict = self._critic_ops.get_critic_state()
                # Sync latest critic to ops
                for ops in self._actor_ops_list:
                    ops.set_critic_state(critic_state_dict)
            else:
                for ops in self._actor_ops_list:
                    ops.update_critic(batch, next_actions)

            # Update actors
            for ops in self._actor_ops_list:
                ops.update_actor(batch)

            # Update version
            self._try_soft_update_target()

    async def train_step_as_task(self) -> None:
        assert not self._params.shared_critic or isinstance(self._critic_ops, RemoteOps)
        assert all(isinstance(ops, RemoteOps) for ops in self._actor_ops_list)
        for _ in range(self._params.num_epoch):
            batch = self._get_batch()
            # Collect next actions
            next_actions = [ops.get_target_action(batch) for ops in self._actor_ops_list]

            # Update critic
            if self._params.shared_critic:
                assert self._critic_ops is not None
                critic_grad = await asyncio.gather(*[self._critic_ops.get_critic_grad(batch, next_actions)])
                assert isinstance(critic_grad, list) and isinstance(critic_grad[0], dict)
                self._critic_ops.update_critic_with_grad(critic_grad[0])
                critic_state_dict = self._critic_ops.get_critic_state()
                # Sync latest critic to ops
                for ops in self._actor_ops_list:
                    ops.set_critic_state(critic_state_dict)
            else:
                critic_grad_list = await asyncio.gather(
                    *[ops.get_critic_grad(batch, next_actions) for ops in self._actor_ops_list]
                )
                for ops, critic_grad in zip(self._actor_ops_list, critic_grad_list):
                    ops.update_critic_with_grad(critic_grad)

            # Update actors
            actor_grad_list = await asyncio.gather(*[ops.get_actor_grad(batch) for ops in self._actor_ops_list])
            for ops, actor_grad in zip(self._actor_ops_list, actor_grad_list):
                ops.update_actor_with_grad(actor_grad)

            # Update version
            self._try_soft_update_target()

    def _try_soft_update_target(self) -> None:
        """Soft update the target policies and target critics."""
        self._policy_version += 1
        if self._policy_version - self._target_policy_version == self._params.update_target_every:
            for ops in self._actor_ops_list:
                ops.soft_update_target()
            if self._params.shared_critic:
                assert self._critic_ops is not None
                self._critic_ops.soft_update_target()
            self._target_policy_version = self._policy_version

    def get_policy_state(self) -> Dict[str, dict]:
        self._assert_ops_exists()
        ret_policy_state = {}
        for ops in self._actor_ops_list:
            policy_name, state = ops.get_policy_state()
            ret_policy_state[policy_name] = state
        return ret_policy_state

    def load(self, path: str) -> None:
        self._assert_ops_exists()

        policy_state_dict = torch.load(os.path.join(path, f"{self.name}_policy.{FILE_SUFFIX}"))
        non_policy_state_dict = torch.load(os.path.join(path, f"{self.name}_non_policy.{FILE_SUFFIX}"))
        for ops_name in policy_state_dict:
            self._ops_dict[ops_name].set_state({**policy_state_dict[ops_name], **non_policy_state_dict[ops_name]})

    def save(self, path: str) -> None:
        self._assert_ops_exists()

        trainer_state = {ops.name: ops.get_state() for ops in self._actor_ops_list}
        if self._params.shared_critic:
            assert self._critic_ops is not None
            trainer_state[self._critic_ops.name] = self._critic_ops.get_state()

        policy_state_dict = {ops_name: state["policy"] for ops_name, state in trainer_state.items()}
        non_policy_state_dict = {ops_name: state["non_policy"] for ops_name, state in trainer_state.items()}

        torch.save(policy_state_dict, os.path.join(path, f"{self.name}_policy.{FILE_SUFFIX}"))
        torch.save(non_policy_state_dict, os.path.join(path, f"{self.name}_non_policy.{FILE_SUFFIX}"))

    def _assert_ops_exists(self) -> None:
        if not self._actor_ops_list:
            raise ValueError("Call 'DiscreteMADDPG.build' to create actor ops first.")
        if self._params.shared_critic and not self._critic_ops:
            raise ValueError("Call 'DiscreteMADDPG.build' to create the critic ops first.")

    async def exit(self) -> None:
        pass
