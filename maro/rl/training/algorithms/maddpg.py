# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from maro.rl.model import MultiQNet
from maro.rl.policy import DiscretePolicyGradient
from maro.rl.rollout import ExpElement
from maro.rl.training import AbsTrainOps, MultiTrainer, RandomMultiReplayMemory, RemoteOps, TrainerParams, remote
from maro.rl.utils import MultiTransitionBatch, ndarray_to_tensor
from maro.utils import clone


@dataclass
class DiscreteMADDPGParams(TrainerParams):
    get_q_critic_net_func: Callable[[], MultiQNet] = None
    num_epoch: int = 10
    update_target_every: int = 5
    soft_update_coef: float = 0.5
    reward_discount: float = 0.9
    q_value_loss_cls: Callable = None
    shared_critic: bool = False

    def __post_init__(self) -> None:
        assert self.get_q_critic_net_func is not None

    def extract_ops_params(self) -> Dict[str, object]:
        return {
            "device": self.device,
            "get_q_critic_net_func": self.get_q_critic_net_func,
            "shared_critic": self.shared_critic,
            "reward_discount": self.reward_discount,
            "soft_update_coef": self.soft_update_coef,
            "update_target_every": self.update_target_every,
            "q_value_loss_func": self.q_value_loss_cls() if self.q_value_loss_cls is not None else torch.nn.MSELoss(),
        }


class DiscreteMADDPGOps(AbsTrainOps):
    def __init__(
        self,
        name: str,
        device: str,
        get_policy_func: Callable[[], DiscretePolicyGradient],
        get_q_critic_net_func: Callable[[], MultiQNet],
        policy_idx: int,
        create_actor: bool,
        *,
        shared_critic: bool = False,
        reward_discount: float = 0.9,
        soft_update_coef: float = 0.5,
        update_target_every: int = 5,
        q_value_loss_func: Callable = None,
    ) -> None:
        super(DiscreteMADDPGOps, self).__init__(
            name=name,
            device=device,
            is_single_scenario=False,
            get_policy_func=get_policy_func
        )

        self._policy_idx = policy_idx
        self._shared_critic = shared_critic

        # Actor
        self._create_actor = create_actor
        if create_actor:
            self._policy = get_policy_func()
            assert isinstance(self._policy, DiscretePolicyGradient)

            self._policy.to_device(self._device)
            self._target_policy: DiscretePolicyGradient = clone(self._policy)
            self._target_policy.set_name(f"target_{self._policy.name}")
            self._target_policy.eval()
            self._target_policy.to_device(self._device)

        # Critic
        self._q_critic_net: MultiQNet = get_q_critic_net_func()
        self._q_critic_net.to(self._device)
        self._target_q_critic_net: MultiQNet = clone(self._q_critic_net)
        self._target_q_critic_net.eval()
        self._target_q_critic_net.to(self._device)

        #
        self._reward_discount = reward_discount
        self._q_value_loss_func = q_value_loss_func
        self._update_target_every = update_target_every
        self._soft_update_coef = soft_update_coef

    def get_target_action(self, batch: MultiTransitionBatch) -> torch.Tensor:
        agent_state = ndarray_to_tensor(batch.agent_states[self._policy_idx], self._device)
        return self._target_policy.get_actions_tensor(agent_state)

    def get_latest_action(self, batch: MultiTransitionBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self._policy, DiscretePolicyGradient)

        agent_state = ndarray_to_tensor(batch.agent_states[self._policy_idx], self._device)
        self._policy.train()
        action = self._policy.get_actions_tensor(agent_state)
        logps = self._policy.get_state_action_logps(agent_state, action)
        return action, logps

    def _get_critic_loss(self, batch: MultiTransitionBatch, next_actions: List[torch.Tensor]) -> torch.Tensor:
        assert not self._shared_critic
        assert isinstance(next_actions, list) and all(isinstance(action, torch.Tensor) for action in next_actions)

        states = ndarray_to_tensor(batch.states, self._device)  # x
        actions = [ndarray_to_tensor(action, self._device) for action in batch.actions]  # a
        next_states = ndarray_to_tensor(batch.next_states, self._device)  # x'
        rewards = ndarray_to_tensor(np.vstack([reward for reward in batch.rewards]), self._device)  # r
        terminals = ndarray_to_tensor(batch.terminals, self._device)  # d

        self._q_critic_net.train()
        with torch.no_grad():
            next_q_values = self._target_q_critic_net.q_values(
                states=next_states,  # x'
                actions=next_actions
            )  # a'
        target_q_values = (
            rewards[self._policy_idx] + self._reward_discount * (1 - terminals.float()) * next_q_values
        )
        q_values = self._q_critic_net.q_values(
            states=states,  # x
            actions=actions  # a
        )  # Q(x, a)
        return self._q_value_loss_func(q_values, target_q_values.detach())

    @remote
    def get_critic_grad(
        self,
        batch: MultiTransitionBatch,
        next_actions: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self._q_critic_net.get_gradients(self._get_critic_loss(batch, next_actions))

    def update_critic(self, batch: MultiTransitionBatch, next_actions: List[torch.Tensor]) -> None:
        self._q_critic_net.train()
        self._q_critic_net.step(self._get_critic_loss(batch, next_actions))

    def update_critic_with_grad(self, grad_dict: dict) -> None:
        self._q_critic_net.train()
        self._q_critic_net.apply_gradients(grad_dict)

    def _get_actor_loss(self, batch: MultiTransitionBatch) -> torch.Tensor:
        latest_action, latest_action_logp = self.get_latest_action(batch)
        states = ndarray_to_tensor(batch.states, self._device)  # x
        actions = [ndarray_to_tensor(action, self._device) for action in batch.actions]  # a
        actions[self._policy_idx] = latest_action
        self._policy.train()
        self._q_critic_net.freeze()
        actor_loss = -(self._q_critic_net.q_values(
            states=states,  # x
            actions=actions  # [a^j_1, ..., a_i, ..., a^j_N]
        ) * latest_action_logp).mean()  # Q(x, a^j_1, ..., a_i, ..., a^j_N)
        self._q_critic_net.unfreeze()
        return actor_loss

    @remote
    def get_actor_grad(self, batch: MultiTransitionBatch) -> Dict[str, torch.Tensor]:
        return self._policy.get_gradients(self._get_actor_loss(batch))

    def update_actor(self, batch: MultiTransitionBatch) -> None:
        self._policy.train()
        self._policy.step(self._get_actor_loss(batch))

    def update_actor_with_grad(self, grad_dict: dict) -> None:
        self._policy.train()
        self._policy.apply_gradients(grad_dict)

    def soft_update_target(self) -> None:
        if self._create_actor:
            self._target_policy.soft_update(self._policy, self._soft_update_coef)
        if not self._shared_critic:
            self._target_q_critic_net.soft_update(self._q_critic_net, self._soft_update_coef)

    def get_critic_state(self) -> dict:
        return {
            "critic": self._q_critic_net.get_state(),
            "target_critic": self._target_q_critic_net.get_state()
        }

    def set_critic_state(self, ops_state_dict: dict) -> dict:
        self._q_critic_net.set_state(ops_state_dict["critic"])
        self._target_q_critic_net.set_state(ops_state_dict["target_critic"])

    def get_actor_state(self) -> dict:
        if self._create_actor:
            return {"policy": self._policy.get_state(), "target_policy": self._target_policy.get_state()}
        else:
            return {}

    def set_actor_state(self, ops_state_dict: dict) -> None:
        if self._create_actor:
            self._policy.set_state(ops_state_dict["policy"])
            self._target_policy.set_state(ops_state_dict["target_policy"])

    def get_state(self) -> dict:
        return {**self.get_actor_state(), **self.get_critic_state()}

    def set_state(self, ops_state_dict: dict) -> None:
        self.set_critic_state(ops_state_dict)
        self.set_actor_state(ops_state_dict)


class DiscreteMADDPG(MultiTrainer):
    def __init__(self, name: str, params: DiscreteMADDPGParams) -> None:
        super(DiscreteMADDPG, self).__init__(name, params)
        self._params = params
        self._ops_params = self._params.extract_ops_params()
        self._state_dim = params.get_q_critic_net_func().state_dim
        self._policy_version = self._target_policy_version = 0
        self._shared_critic_ops_name = f"{self._name}.shared_critic_ops"

        self._actor_ops_list = []
        self._actor_ops_dict = {}
        self._critic_ops = None
        self._replay_memory = None
        self._policy2agent = {}

    def build(self) -> None:
        self._actor_ops_list = [self.get_ops(f"{self._name}.actor_{i}_ops") for i in range(len(self._policy_names))]
        self._actor_ops_dict = {ops.name: ops for ops in self._actor_ops_list}
        if self._params.shared_critic:
            self._critic_ops = self.get_ops(self._shared_critic_ops_name)
        else:
            self._critic_ops = None

        self._replay_memory = RandomMultiReplayMemory(
            capacity=self._params.replay_memory_capacity,
            state_dim=self._state_dim,
            action_dims=[ops.policy_action_dim for ops in self._actor_ops_list],
            agent_states_dims=[ops.policy_state_dim for ops in self._actor_ops_list]
        )

        assert len(self._agent2policy.keys()) == len(self._agent2policy.values())  # agent <=> policy
        self._policy2agent = {policy_name: agent_name for agent_name, policy_name in self._agent2policy.items()}

    def record(self, env_idx: int, exp_element: ExpElement) -> None:
        assert exp_element.num_agents == len(self._agent2policy.keys())

        if min(exp_element.terminal_dict.values()) != max(exp_element.terminal_dict.values()):
            raise ValueError("The 'terminal` flag of all agents must be identical.")
        terminal_flag = min(exp_element.terminal_dict.values())

        actions = []
        rewards = []
        agent_states = []
        next_agent_states = []
        for policy_name in self._policy_names:
            agent_name = self._policy2agent[policy_name]
            actions.append(np.expand_dims(exp_element.action_dict[agent_name], axis=0))
            rewards.append(np.array([exp_element.reward_dict[agent_name]]))
            agent_states.append(np.expand_dims(exp_element.agent_state_dict[agent_name], axis=0))
            next_agent_states.append(np.expand_dims(
                exp_element.next_agent_state_dict.get(agent_name, exp_element.agent_state_dict[agent_name]), axis=0
            ))

        transition_batch = MultiTransitionBatch(
            states=np.expand_dims(exp_element.state, axis=0),
            actions=actions,
            rewards=rewards,
            next_states=np.expand_dims(
                exp_element.next_state if exp_element.next_state is not None else exp_element.state, axis=0
            ),
            agent_states=agent_states,
            next_agent_states=next_agent_states,
            terminals=np.array([terminal_flag])
        )
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> MultiTransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._batch_size)

    def get_local_ops_by_name(self, name: str) -> AbsTrainOps:
        if name == self._shared_critic_ops_name:
            ops_params = dict(self._ops_params)
            ops_params.update({
                "get_policy_func": None,
                "policy_idx": -1,
                "shared_critic": False,
                "create_actor": False,
            })
            return DiscreteMADDPGOps(name=name, **ops_params)
        else:
            policy_idx = self.get_policy_idx_from_ops_name(name)
            policy_name = self._policy_names[policy_idx]

            ops_params = dict(self._ops_params)
            ops_params.update({
                "get_policy_func": lambda: self._policy_creator[policy_name](policy_name),
                "policy_idx": policy_idx,
                "create_actor": True,
            })
            return DiscreteMADDPGOps(name=name, **ops_params)

    def train(self):
        assert not self._params.shared_critic or isinstance(self._critic_ops, DiscreteMADDPGOps)
        assert all(isinstance(ops, DiscreteMADDPGOps) for ops in self._actor_ops_list)
        for _ in range(self._params.num_epoch):
            batch = self._get_batch()
            # Collect next actions
            next_actions = [ops.get_target_action(batch) for ops in self._actor_ops_list]

            # Update critic
            if self._params.shared_critic:
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
            self._policy_version += 1
            self._try_soft_update_target()

    async def train_as_task(self):
        assert not self._params.shared_critic or isinstance(self._critic_ops, RemoteOps)
        assert all(isinstance(ops, RemoteOps) for ops in self._actor_ops_list)
        for _ in range(self._params.num_epoch):
            batch = self._get_batch()
            # Collect next actions
            next_actions = [ops.get_target_action(batch) for ops in self._actor_ops_list]

            # Update critic
            if self._params.shared_critic:
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
            self._policy_version += 1
            self._try_soft_update_target()

    def _try_soft_update_target(self) -> None:
        if self._policy_version - self._target_policy_version == self._params.update_target_every:
            for ops in self._actor_ops_list:
                ops.soft_update_target()
            if self._params.shared_critic:
                self._critic_ops.soft_update_target()
            self._target_policy_version = self._policy_version

    def get_policy_state(self) -> Dict[str, object]:
        self._assert_ops_exists()
        ret_policy_state = {}
        for ops in self._actor_ops_list:
            policy_name, state = ops.get_policy_state()
            ret_policy_state[policy_name] = state
        return ret_policy_state

    def load(self, path: str):
        self._assert_ops_exists()
        trainer_state = torch.load(path)
        for ops_name, ops_state in trainer_state.items():
            if ops_name == self._critic_ops.name:
                self._critic_ops.set_state(ops_state)
            else:
                self._actor_ops_dict[ops_name].set_state(torch.load(path))

    def save(self, path: str):
        self._assert_ops_exists()
        trainer_state = {ops.name: ops.get_state() for ops in self._actor_ops_list}
        if self._params.shared_critic:
            trainer_state[self._critic_ops.name] = self._critic_ops.get_state()
        torch.save(trainer_state, path)

    def _assert_ops_exists(self):
        if not self._actor_ops_list:
            raise ValueError("Call 'DiscreteMADDPG.build' to create actor ops first.")
        if self._params.shared_critic and not self._critic_ops:
            raise ValueError("Call 'DiscreteMADDPG.build' to create the critic ops first.")

    @staticmethod
    def get_policy_idx_from_ops_name(ops_name):
        _, sub_name = ops_name.split(".")
        return int(sub_name.split("_")[1])
