# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch

from maro.rl_v3.model import MultiQNet
from maro.rl_v3.policy import DiscretePolicyGradient
from maro.rl_v3.training import AbsTrainOps, MultiTrainer, RandomMultiReplayMemory, TrainerParams
from maro.rl_v3.utils import MultiTransitionBatch, RemoteObj, ndarray_to_tensor
from maro.utils import clone


@dataclass
class DiscreteMADDPGParams(TrainerParams):
    get_q_critic_net_func: Callable[[], MultiQNet] = None
    num_epoch: int = 10
    update_target_every: int = 5
    soft_update_coef: float = 0.5
    reward_discount: float = 0.9
    q_value_loss_cls: Callable = None
    critic_loss_coef: float = 1.0
    shared_critic: bool = False

    def __post_init__(self) -> None:
        assert self.get_q_critic_net_func is not None

    def extract_ops_params(self) -> Dict[str, object]:
        return {
            "device": self.device,
            "enable_data_parallelism": self.enable_data_parallelism,
            "get_q_critic_net_func": self.get_q_critic_net_func,
            "shared_critic": self.shared_critic,
            "reward_discount": self.reward_discount,
            "critic_loss_coef": self.critic_loss_coef,
            "soft_update_coef": self.soft_update_coef,
            "update_target_every": self.update_target_every,
            "q_value_loss_func": self.q_value_loss_cls() if self.q_value_loss_cls is not None else torch.nn.MSELoss(),
        }


class DiscreteMADDPGOps(AbsTrainOps):
    def __init__(
        self,
        device: str,
        get_policy_func: Callable[[], DiscretePolicyGradient],
        get_q_critic_net_func: Callable[[], MultiQNet],
        agent_idx: int,
        create_actor: bool,
        enable_data_parallelism: bool = False,
        *,
        shared_critic: bool = False,
        reward_discount: float = 0.9,
        critic_loss_coef: float = 1.0,
        soft_update_coef: float = 0.5,
        update_target_every: int = 5,
        q_value_loss_func: Callable = None,
    ) -> None:
        super(DiscreteMADDPGOps, self).__init__(
            device=device,
            is_single_scenario=False,
            get_policy_func=get_policy_func,
            enable_data_parallelism=enable_data_parallelism
        )

        self._agent_idx = agent_idx
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
        self._critic_loss_coef = critic_loss_coef
        self._q_value_loss_func = q_value_loss_func
        self._update_target_every = update_target_every
        self._soft_update_coef = soft_update_coef

    def get_target_action(self) -> torch.Tensor:
        agent_state = ndarray_to_tensor(self._batch.agent_states[self._agent_idx], self._device)
        return self._target_policy.get_actions_tensor(agent_state)

    def get_latest_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self._policy, DiscretePolicyGradient)

        agent_state = ndarray_to_tensor(self._batch.agent_states[self._agent_idx], self._device)
        self._policy.train()
        action = self._policy.get_actions_tensor(agent_state)
        logps = self._policy.get_state_action_logps(agent_state, action)
        return action, logps

    def get_state_dict(self, scope: str = "all") -> dict:
        ret_dict = {}
        if scope in ("all", "actor"):
            ret_dict["policy_state"] = self._policy.get_policy_state()
            ret_dict["target_policy_state"] = self._target_policy.get_policy_state()
        if scope in ("all", "critic"):
            ret_dict["critic_state"] = self._q_critic_net.get_net_state()
            ret_dict["target_critic_state"] = self._target_q_critic_net.get_net_state()
        return ret_dict

    def set_state_dict(self, ops_state_dict: dict, scope: str = "all") -> None:
        if scope in ("all", "actor"):
            self._policy.set_policy_state(ops_state_dict["policy_state"])
            self._target_policy.set_policy_state(ops_state_dict["target_policy_state"])
        if scope in ("all", "critic"):
            self._q_critic_net.set_net_state(ops_state_dict["critic_state"])
            self._target_q_critic_net.set_net_state(ops_state_dict["target_critic_state"])

    def _get_critic_grad(
        self,
        batch: MultiTransitionBatch,
        next_actions: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        assert not self._shared_critic

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
            rewards[self._agent_idx] + self._reward_discount * (1 - terminals.float()) * next_q_values
        )
        q_values = self._q_critic_net.q_values(
            states=states,  # x
            actions=actions  # a
        )  # Q(x, a)
        critic_loss = self._q_value_loss_func(q_values, target_q_values.detach()) * self._critic_loss_coef
        return self._q_critic_net.get_gradients(critic_loss)

    def _get_actor_grad(
        self,
        batch: MultiTransitionBatch,
        latest_action: torch.Tensor,
        latest_action_logp: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        states = ndarray_to_tensor(batch.states, self._device)  # x
        actions = [ndarray_to_tensor(action, self._device) for action in batch.actions]  # a
        actions[self._agent_idx] = latest_action

        self._policy.train()
        self._q_critic_net.freeze()
        actor_loss = -(self._q_critic_net.q_values(
            states=states,  # x
            actions=actions  # [a^j_1, ..., a_i, ..., a^j_N]
        ) * latest_action_logp).mean()  # Q(x, a^j_1, ..., a_i, ..., a^j_N)
        self._q_critic_net.unfreeze()
        return self._policy.get_gradients(actor_loss)

    def get_batch_grad(
        self,
        batch: MultiTransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        assert scope in ("all", "actor", "critic"), \
            f"Unrecognized scope {scope}. Excepting 'all', 'actor', or 'critic'."

        if tensor_dict is None:
            tensor_dict = {}

        grad_dict = {}
        if scope in ("all", "critic"):
            next_actions = tensor_dict["next_actions"]
            assert isinstance(next_actions, list)
            assert all(isinstance(action, torch.Tensor) for action in next_actions)

            grad_dict["critic_grads"] = self._get_critic_grad(batch, next_actions)
        if scope in ("all", "actor"):
            latest_action = tensor_dict["latest_action"]
            latest_action_logp = tensor_dict["latest_action_logp"]
            assert isinstance(latest_action, torch.Tensor)
            assert isinstance(latest_action_logp, torch.Tensor)

            grad_dict["actor_grads"] = self._get_actor_grad(batch, latest_action, latest_action_logp)

        return grad_dict

    def _dispatch_tensor_dict(self, tensor_dict: Dict[str, object], num_ops: int) -> List[Dict[str, object]]:
        raise NotImplementedError

    def _dispatch_batch(self, batch: MultiTransitionBatch, num_ops: int) -> List[MultiTransitionBatch]:
        # batch_size = batch.states.shape[0]
        # assert batch_size >= num_ops, \
        #     f"Batch size should be greater than or equal to num_ops, but got {batch_size} and {num_ops}."
        # sub_batch_indexes = [range(batch_size)[i::num_ops] for i in range(num_ops)]
        # sub_batches = [MultiTransitionBatch(
        #     policy_names=[],
        #     states=batch.states[indexes],
        #     actions=[action[indexes] for action in batch.actions],
        #     rewards=[reward[indexes] for reward in batch.rewards],
        #     terminals=batch.terminals[indexes],
        #     next_states=batch.next_states[indexes],
        #     agent_states=[state[indexes] for state in batch.agent_states],
        #     next_agent_states=[state[indexes] for state in batch.next_agent_states]
        # ) for indexes in sub_batch_indexes]
        # return sub_batches
        raise NotImplementedError

    def update_critic(self, next_actions: List[torch.Tensor]) -> None:
        assert not self._shared_critic

        grads = self._get_batch_grad(
            self._batch,
            tensor_dict={"next_actions": next_actions},
            scope="critic"
        )

        self._q_critic_net.train()
        self._q_critic_net.apply_gradients(grads["critic_grads"])

    def update_actor(self, latest_action: torch.Tensor, latest_action_logp: torch.Tensor) -> None:
        grads = self._get_batch_grad(
            self._batch,
            tensor_dict={
                "latest_action": latest_action,
                "latest_action_logp": latest_action_logp
            },
            scope="actor"
        )

        self._policy.train()
        self._policy.apply_gradients(grads["actor_grads"])

    def soft_update_target(self) -> None:
        if self._create_actor:
            self._target_policy.soft_update(self._policy, self._soft_update_coef)
        if not self._shared_critic:
            self._target_q_critic_net.soft_update(self._q_critic_net, self._soft_update_coef)


class DiscreteMADDPG(MultiTrainer):
    def __init__(self, name: str, params: DiscreteMADDPGParams) -> None:
        super(DiscreteMADDPG, self).__init__(name, params)

        self._params = params

        self._state_dim = params.get_q_critic_net_func().state_dim

        self._policy_version = self._target_policy_version = 0

    def _improve(self, batch: MultiTransitionBatch) -> None:
        for ops in self._actor_ops_list:
            ops.set_batch(batch)

        # Collect next actions
        next_actions: List[torch.Tensor] = []
        for i, ops in enumerate(self._actor_ops_list):
            next_actions.append(ops.get_target_action())

        # Update critic
        if self._params.shared_critic:
            self._critic_ops.set_batch(batch)
            self._critic_ops.update_critic(next_actions=next_actions)
            critic_state_dict = self._critic_ops.get_state_dict(scope="critic")

            # Sync latest critic to ops
            for ops in self._actor_ops_list:
                ops.set_state_dict(critic_state_dict, scope="critic")
        else:
            for ops in self._actor_ops_list:
                ops.update_critic(next_actions=next_actions)

        # Update actor
        latest_actions: List[torch.Tensor] = []
        latest_action_logps: List[torch.Tensor] = []
        for i, ops in enumerate(self._actor_ops_list):
            cur_action, cur_logps = ops.get_latest_action()
            latest_actions.append(cur_action)
            latest_action_logps.append(cur_logps)

        for i, ops in enumerate(self._actor_ops_list):
            ops.update_actor(latest_actions[i], latest_action_logps[i])

        # Update version
        self._try_soft_update_target()

    def build(self) -> None:
        self._actor_ops_list: List[Union[RemoteObj, AbsTrainOps]] = []
        for i, policy_name in enumerate(self._policy_names):
            ops_name = f"ops_{i}"
            ops = self.get_ops(ops_name)
            self._actor_ops_list.append(ops)

        if self._params.shared_critic:
            ops_name = "critic_ops"
            ops = self.get_ops(ops_name)
            self._critic_ops = ops

        self._replay_memory = RandomMultiReplayMemory(
            capacity=self._params.replay_memory_capacity,
            state_dim=self._state_dim,
            action_dims=[ops.policy_action_dim for ops in self._actor_ops_list],
            agent_states_dims=[ops.policy_state_dim for ops in self._actor_ops_list]
        )

    def _get_local_ops_by_name(self, ops_name: str) -> AbsTrainOps:
        if ops_name == "critic_ops":
            cur_ops_params = {
                "get_policy_func": None,
                "agent_idx": -1,
                **self._params.extract_ops_params(),
                "shared_critic": False,
                "create_actor": False,
            }

            return DiscreteMADDPGOps(**cur_ops_params)
        else:
            i = int(ops_name.split("_")[1])
            policy_name = self._policy_names[i]

            cur_ops_params = {
                "get_policy_func": lambda: self._get_policy_func_dict[policy_name](policy_name),
                "agent_idx": i,
                **self._params.extract_ops_params(),
                "create_actor": True,
            }
            return DiscreteMADDPGOps(**cur_ops_params)

    async def train_step(self):
        for _ in range(self._params.num_epoch):
            batch = self._get_batch()
            await asyncio.gather(*[ops.set_batch(batch) for ops in self._actor_ops_list])

            # Collect next actions
            next_actions = await asyncio.gather(*[ops.get_target_action() for ops in self._actor_ops_list])
            assert isinstance(next_actions, list) and all(isinstance(action, torch.Tensor) for action in next_actions)

            # Update critic
            if self._params.shared_critic:
                await asyncio.gather(self._critic_ops.set_batch(batch))
                await asyncio.gather(self._critic_ops.update_critic(next_actions))
                critic_state_dict = await asyncio.gather(self._critic_ops.get_state_dict(scope="critic"))
                assert isinstance(critic_state_dict, list) and len(critic_state_dict) == 1

                # Sync latest critic to ops
                await asyncio.gather(*[
                    ops.set_state_dict(critic_state_dict[0], scope="critic") for ops in self._actor_ops_list
                ])
            else:
                await asyncio.gather(*[ops.update_critic(next_actions) for ops in self._actor_ops_list])

            # Update actor
            latest_action_info = await asyncio.gather(*[ops.get_latest_action() for ops in self._actor_ops_list])
            await asyncio.gather(*[
                ops.update_actor(*info) for ops, info in zip(self._actor_ops_list, latest_action_info)
            ])

            # Update version
            await self._try_soft_update_target()

    async def _try_soft_update_target(self) -> None:
        self._policy_version += 1
        if self._policy_version - self._target_policy_version == self._params.update_target_every:
            parallel_updates = [ops.soft_update_target() for ops in self._actor_ops_list]
            if self._params.shared_critic:
                parallel_updates.append(self._critic_ops.soft_update_target())
            await asyncio.gather(*parallel_updates)
            self._target_policy_version = self._policy_version

    def get_policy_state_dict(self) -> Dict[str, object]:
        if self.num_policies == 0:
            raise ValueError("'create_ops' needs to be called to create an ops instance first.")

        return {ops.policy_name: ops.get_policy_state() for ops in self._actor_ops_list}
