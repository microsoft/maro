from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from maro.rl_v3.model import MultiQNet
from maro.rl_v3.policy import DiscretePolicyGradient, RLPolicy
from maro.rl_v3.policy_trainer import MultiTrainer
from maro.rl_v3.replay_memory import RandomMultiReplayMemory
from maro.rl_v3.utils import MultiTransitionBatch, ndarray_to_tensor
from maro.utils import clone


class DiscreteMADDPGWorker(object):
    def __init__(
        self,
        reward_discount: float,
        get_q_critic_net_func: Callable[[], MultiQNet],
        shared_critic: bool = False,
        device: str = None,
        data_parallel: bool = False,
        critic_loss_coef: float = 1.0,
        soft_update_coef: float = 0.5,
        update_target_every: int = 5,
        q_value_loss_func: Callable = None
    ) -> None:
        super(DiscreteMADDPGWorker, self).__init__()

        self._device = torch.device(device) if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor
        self._policies: Optional[Dict[int, DiscretePolicyGradient]] = None
        self._target_policies: Optional[Dict[int, DiscretePolicyGradient]] = None

        # Critic
        self._get_q_critic_net_func = get_q_critic_net_func
        self._q_critic_nets: Optional[Dict[int, MultiQNet]] = None
        self._target_q_critic_nets: Optional[Dict[int, MultiQNet]] = None

        #
        self._shared_critic = shared_critic
        self._num_policies = 0
        self._indexes: List[int] = []

        self._batch: Optional[MultiTransitionBatch] = None
        self._policy_version: int = 0
        self._target_policy_version: int = 0

        self._data_parallel = data_parallel
        self._reward_discount = reward_discount
        self._critic_loss_coef = critic_loss_coef
        self._q_value_loss_func = q_value_loss_func
        self._update_target_every = update_target_every
        self._soft_update_coef = soft_update_coef

    def register_policies(self, policy_dict: Dict[int, RLPolicy]) -> None:
        self._register_policies_impl(policy_dict)

    def _register_policies_impl(self, policy_dict: Dict[int, RLPolicy]) -> None:
        self._num_policies = len(policy_dict)
        self._indexes = sorted(policy_dict.keys())

        # Actors
        self._policies: Dict[int, DiscretePolicyGradient] = {}
        self._target_policies: Dict[int, DiscretePolicyGradient] = {}
        for i, policy in policy_dict.items():
            assert isinstance(policy, DiscretePolicyGradient)
            target_policy: DiscretePolicyGradient = clone(policy)
            target_policy.set_name(f"target_{policy.name}")
            target_policy.to_device(self._device)
            target_policy.eval()

            self._policies[i] = policy
            self._target_policies[i] = target_policy

        # Critic
        self._q_critic_nets: Dict[int, MultiQNet] = {}
        self._target_q_critic_nets: Dict[int, MultiQNet] = {}
        indexes = [0] if self._shared_critic else self._indexes
        for i in indexes:
            q_critic_net = self._get_q_critic_net_func()
            q_critic_net.to(self._device)
            target_q_critic_net = clone(q_critic_net)
            target_q_critic_net.to(self._device)
            target_q_critic_net.eval()
            self._q_critic_nets[i] = q_critic_net
            self._target_q_critic_nets[i] = target_q_critic_net

    def set_batch(self, batch: MultiTransitionBatch) -> None:
        self._batch = batch

    def get_target_action_dict(self) -> Dict[int, torch.Tensor]:
        agent_state_dict = {
            i: ndarray_to_tensor(self._batch.agent_states[i], self._device)
            for i in self._indexes
        }  # o
        with torch.no_grad():
            action_dict = {
                i: policy.get_actions_tensor(agent_state_dict[i])
                for i, policy in self._target_policies.items()
            }
        return action_dict

    def get_latest_action_dict(self) -> Tuple[dict, dict]:
        agent_state_dict = {
            i: ndarray_to_tensor(self._batch.agent_states[i], self._device)
            for i in self._indexes
        }  # o

        latest_actions = {}
        latest_action_logps = {}
        for i, policy in self._policies.items():
            policy.train()
            action = policy.get_actions_tensor(agent_state_dict[i])
            logps = policy.get_state_action_logps(agent_state_dict[i], action)
            latest_actions[i] = action
            latest_action_logps[i] = logps

        return latest_actions, latest_action_logps

    def _get_batch_grad(
        self,
        batch: MultiTransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
        if self._data_parallel:
            raise NotImplementedError  # TODO
        else:
            return self.get_batch_grad(batch, tensor_dict, scope)

    def _get_critic_grad(
        self,
        batch: MultiTransitionBatch,
        next_actions: List[torch.Tensor]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        states = ndarray_to_tensor(batch.states, self._device)  # x
        actions = [ndarray_to_tensor(action, self._device) for action in batch.actions]  # a

        next_states = ndarray_to_tensor(batch.next_states, self._device)  # x'
        rewards = ndarray_to_tensor(np.vstack([reward for reward in batch.rewards]), self._device)  # r
        terminals = ndarray_to_tensor(batch.terminals, self._device)  # d

        for net in self._q_critic_nets.values():
            net.train()

        critic_loss_dict = {}
        for i in self._indexes:
            q_net = self._q_critic_nets[i]
            target_q_net = self._target_q_critic_nets[i]
            with torch.no_grad():
                next_q_values = target_q_net.q_values(
                    states=next_states,  # x'
                    actions=next_actions
                )  # a'
            target_q_values = (rewards[i] + self._reward_discount * (1 - terminals.float()) * next_q_values)
            q_values = q_net.q_values(
                states=states,  # x
                actions=actions  # a
            )  # Q(x, a)
            critic_loss = self._q_value_loss_func(q_values, target_q_values.detach()) * self._critic_loss_coef
            critic_loss_dict[i] = critic_loss

        return {
            i: self._q_critic_nets[i].get_gradients(critic_loss_dict[i])
            for i in self._indexes
        }

    def _get_actor_grad(
        self,
        batch: MultiTransitionBatch,
        latest_actions: List[torch.Tensor],
        latest_action_logps: List[torch.Tensor]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        states = ndarray_to_tensor(batch.states, self._device)  # x
        actions = [ndarray_to_tensor(action, self._device) for action in batch.actions]  # a

        for policy in self._policies.values():
            policy.train()

        actor_loss_dict = {}
        for i in self._indexes:
            q_net = self._q_critic_nets[i]
            q_net.freeze()

            action_backup = actions[i]
            actions[i] = latest_actions[i]  # Replace latest action
            actor_loss = -(q_net.q_values(
                states=states,  # x
                actions=actions  # [a^j_1, ..., a_i, ..., a^j_N]
            ) * latest_action_logps[i]).mean()  # Q(x, a^j_1, ..., a_i, ..., a^j_N)
            actor_loss_dict[i] = actor_loss

            actions[i] = action_backup  # Restore original action
            q_net.unfreeze()

        return {
            i: self._policies[i].get_gradients(actor_loss_dict[i])
            for i in self._indexes
        }

    def get_batch_grad(
        self,
        batch: MultiTransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
        assert scope in ("all", "actor", "critic"), \
            f"Unrecognized scope {scope}. Excepting 'all', 'actor', or 'critic'."

        if tensor_dict is None:
            tensor_dict = {}

        grad_dict = {}
        if scope in ("all", "critic"):
            assert "next_actions" in tensor_dict
            next_actions = tensor_dict["next_actions"]
            assert isinstance(next_actions, list)
            assert all(isinstance(action, torch.Tensor) for action in next_actions)

            grad_dict["critic_grads"] = self._get_critic_grad(batch, next_actions)
        if scope in ("all", "actor"):
            assert "latest_actions" in tensor_dict
            assert "latest_action_logps" in tensor_dict
            latest_actions = tensor_dict["latest_actions"]
            latest_action_logps = tensor_dict["latest_action_logps"]
            assert isinstance(latest_actions, list) and isinstance(latest_action_logps, list)
            assert all(isinstance(action, torch.Tensor) for action in latest_actions)
            assert all(isinstance(logps, torch.Tensor) for logps in latest_action_logps)

            grad_dict["actor_grads"] = self._get_actor_grad(batch, latest_actions, latest_action_logps)

        return grad_dict

    def update_critics(self, next_actions: List[torch.Tensor]) -> None:
        assert not self._shared_critic

        grads = self._get_batch_grad(
            self._batch,
            tensor_dict={"next_actions": next_actions},
            scope="critic"
        )

        for i, grad in grads["critic_grads"].items():
            self._q_critic_nets[i].train()
            self._q_critic_nets[i].apply_gradients(grad)

    def set_critic_state(self, net_state: object, target_net_state: object) -> None:
        assert self._shared_critic

        self._q_critic_nets[0].set_net_state(net_state)
        self._target_q_critic_nets[0].set_net_state(target_net_state)

    def get_critic_state(self) -> tuple:
        return self._q_critic_nets[0].get_net_state(), self._target_q_critic_nets[0].get_net_state()

    def update_actors(self, latest_actions: List[torch.Tensor], latest_action_logps: List[torch.Tensor]) -> None:
        grads = self._get_batch_grad(
            self._batch,
            tensor_dict={
                "latest_actions": latest_actions,
                "latest_action_logps": latest_action_logps
            },
            scope="actor"
        )

        for i, grad in grads["actor_grads"].items():
            self._policies[i].train()
            self._policies[i].apply_gradients(grad)

    def update_target_policy(self) -> None:
        self._policy_version += 1
        if self._policy_version - self._target_policy_version == self._update_target_every:
            for i in self._indexes:
                self._target_policies[i].soft_update(self._policies[i], self._soft_update_coef)
                if not self._shared_critic:
                    self._target_q_critic_nets[i].soft_update(self._q_critic_nets[i], self._soft_update_coef)
            self._target_policy_version = self._policy_version

    def get_policy_state_dict(self) -> Dict[int, object]:
        return {
            i: self._policies[i].get_policy_state()
            for i in self._indexes
        }

    def set_policy_state_dict(self, policy_state_dict: Dict[int, object]) -> None:
        for i, policy_state in policy_state_dict.items():
            self._policies[i].set_policy_state(policy_state)

    def get_trainer_state_dict(self) -> dict:
        return {
            "policy_state": {i: self._policies[i].get_policy_state() for i in self._indexes},
            "target_policy_state": {i: self._target_policies[i].get_policy_state() for i in self._indexes},
            "critic_state": {i: self._q_critic_nets[i].get_net_state() for i in self._indexes},
            "target_critic_state": {i: self._target_q_critic_nets[i].get_net_state() for i in self._indexes}
        }

    def set_trainer_state_dict(self, trainer_state_dict: dict) -> None:
        for i in self._indexes:
            self._policies[i].set_policy_state(trainer_state_dict["policy_state"][i])
            self._target_policies[i].set_policy_state(trainer_state_dict["target_policy_state"][i])
            self._q_critic_nets[i].set_net_state(trainer_state_dict["critic_state"][i])
            self._target_q_critic_nets[i].set_net_state(trainer_state_dict["target_critic_state"][i])


class DistributedDiscreteMADDPG(MultiTrainer):
    def __init__(
        self,
        name: str,
        reward_discount: float,
        get_q_critic_net_func: Callable[[], MultiQNet],
        group_size: int = 1,
        policies: List[RLPolicy] = None,
        replay_memory_capacity: int = 10000,
        num_epoch: int = 10,
        update_target_every: int = 5,
        soft_update_coef: float = 0.5,
        train_batch_size: int = 32,
        q_value_loss_cls: Callable = None,
        device: str = None,
        critic_loss_coef: float = 1.0,
        shared_critic: bool = False,
        data_parallel: bool = False
    ) -> None:
        super(DistributedDiscreteMADDPG, self).__init__(name, device, data_parallel)

        self._workers: Optional[List[DiscreteMADDPGWorker]] = None
        self._worker_indexes: Optional[List[List[int]]] = None
        self._num_policies = 0
        self._device_str = device

        self._get_q_critic_net_func = get_q_critic_net_func
        self._q_critic_net: Optional[MultiQNet] = None
        self._target_q_critic_net: Optional[MultiQNet] = None
        self._group_size = group_size
        self._replay_memory_capacity = replay_memory_capacity
        self._target_policies: Optional[List[DiscretePolicyGradient]] = None
        self._shared_critic = shared_critic
        self._policy_names = Optional[List[str]]
        if policies is not None:
            self.register_policies(policies)

        self._num_epoch = num_epoch
        self._update_target_every = update_target_every
        self._policy_version = self._target_policy_version = 0
        self._soft_update_coef = soft_update_coef
        self._train_batch_size = train_batch_size
        self._reward_discount = reward_discount
        self._critic_loss_coef = critic_loss_coef

        self._q_value_loss_func = q_value_loss_cls() if q_value_loss_cls is not None else torch.nn.MSELoss()

    def _get_num_policies(self) -> int:
        return self._num_policies

    def _record_impl(self, transition_batch: MultiTransitionBatch) -> None:
        self._replay_memory.put(transition_batch)

    def _register_policies_impl(self, policies: List[RLPolicy]) -> None:
        assert all(isinstance(policy, DiscretePolicyGradient) for policy in policies)

        self._num_policies = len(policies)
        self._policy_names = [policy.name for policy in policies]

        if self._shared_critic:
            self._q_critic_net = self._get_q_critic_net_func()
            self._q_critic_net.to(self._device)
            self._target_q_critic_net = clone(self._q_critic_net)
            self._target_q_critic_net.to(self._device)
            self._target_q_critic_net.eval()
        else:
            self._q_critic_net = None
            self._target_q_critic_net = None

        self._workers: List[DiscreteMADDPGWorker] = []
        self._worker_indexes: List[List[int]] = []
        cursor = 0
        while cursor < self.num_policies:
            worker = DiscreteMADDPGWorker(
                reward_discount=self._reward_discount, get_q_critic_net_func=self._get_q_critic_net_func,
                shared_critic=self._shared_critic, device=self._device_str, data_parallel=self._data_parallel,
                critic_loss_coef=self._critic_loss_coef, soft_update_coef=self._soft_update_coef,
                update_target_every=self._update_target_every, q_value_loss_func=self._q_value_loss_func
            )

            cursor_end = min(cursor + self._group_size, self.num_policies)
            indexes = list(range(cursor, cursor_end))
            cursor = cursor_end
            worker.register_policies({i: policies[i] for i in indexes})

            self._workers.append(worker)
            self._worker_indexes.append(indexes)

        # Replay
        self._replay_memory = RandomMultiReplayMemory(
            capacity=self._replay_memory_capacity,
            state_dim=self._get_q_critic_net_func().state_dim,
            action_dims=[policy.action_dim for policy in policies],
            agent_states_dims=[policy.state_dim for policy in policies]
        )

    def _get_batch(self, batch_size: int = None) -> MultiTransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def _train_step_impl(self) -> None:
        for _ in range(self._num_epoch):
            self._improve(self._get_batch())

    def _improve(self, batch: MultiTransitionBatch) -> None:
        for worker in self._workers:
            worker.set_batch(batch)

        # Collect next actions
        next_action_dict: Dict[int, torch.Tensor] = {}
        for worker in self._workers:
            next_action_dict.update(worker.get_target_action_dict())
        next_actions = [next_action_dict[i] for i in range(self.num_policies)]

        # Update critic
        if self._shared_critic:
            grads = self._get_batch_grad(
                batch,
                tensor_dict={"next_actions": next_actions},
                scope="critic"
            )
            self._q_critic_net.train()
            self._q_critic_net.apply_gradients(grads["critic_grads"][0])

            # Sync latest critic to workers
            for worker in self._workers:
                worker.set_critic_state(
                    net_state=self._q_critic_net.get_net_state(),
                    target_net_state=self._target_q_critic_net.get_net_state()
                )
        else:
            for worker in self._workers:
                worker.update_critics(next_actions=next_actions)

        # Update actor
        latest_actions_dict = {}
        latest_action_logps_dict = {}
        for worker in self._workers:
            cur_action_dict, cur_logps_dict = worker.get_latest_action_dict()
            latest_actions_dict.update(cur_action_dict)
            latest_action_logps_dict.update(cur_logps_dict)
        latest_actions = [latest_actions_dict[i] for i in range(self.num_policies)]
        latest_action_logps = [latest_action_logps_dict[i] for i in range(self.num_policies)]

        for worker in self._workers:
            worker.update_actors(latest_actions, latest_action_logps)

        # Update version
        self._update_target_policy()

    def _get_critic_grad(
        self,
        batch: MultiTransitionBatch,
        tensor_dict: Dict[str, object]
    ) -> List[Dict[str, torch.Tensor]]:
        assert self._shared_critic

        states = ndarray_to_tensor(batch.states, self._device)  # x
        actions = [ndarray_to_tensor(action, self._device) for action in batch.actions]  # a

        next_actions = tensor_dict["next_actions"]
        assert isinstance(next_actions, list)

        next_states = ndarray_to_tensor(batch.next_states, self._device)  # x'
        rewards = ndarray_to_tensor(np.vstack([reward for reward in batch.rewards]), self._device)  # r
        terminals = ndarray_to_tensor(batch.terminals, self._device)  # d

        self._q_critic_net.train()

        with torch.no_grad():
            next_q_values = self._target_q_critic_net.q_values(
                states=next_states,  # x'
                actions=next_actions  # a'
            )  # Q'(x', a')
        # sum(rewards) for shard critic
        target_q_values = (rewards.sum(0) + self._reward_discount * (1 - terminals.float()) * next_q_values)
        q_values = self._q_critic_net.q_values(
            states=states,  # x
            actions=actions  # a
        )  # Q(x, a)
        critic_loss = self._q_value_loss_func(q_values, target_q_values.detach()) * self._critic_loss_coef

        return [self._q_critic_net.get_gradients(critic_loss)]

    def get_batch_grad(
        self,
        batch: MultiTransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        assert scope in ("all", "critic"), \
            f"Unrecognized scope {scope}. Excepting 'all', 'critic'."

        grad_dict = {}
        if scope in ("all", "critic"):
            grad_dict["critic_grads"] = self._get_critic_grad(batch, tensor_dict)

        return grad_dict

    def _update_target_policy(self) -> None:
        self._policy_version += 1
        if self._policy_version - self._target_policy_version == self._update_target_every:
            if self._shared_critic:
                self._target_q_critic_net.soft_update(self._q_critic_net, self._soft_update_coef)
            self._target_policy_version = self._policy_version

        for worker in self._workers:
            worker.update_target_policy()

    def get_trainer_state_dict(self) -> dict:
        raise NotImplementedError

    def set_trainer_state_dict(self, trainer_state_dict: dict) -> None:
        raise NotImplementedError

    def get_policy_state_dict(self) -> Dict[str, object]:
        policy_state_dict = {}
        for worker in self._workers:
            policy_state_dict.update(worker.get_policy_state_dict())
        return {name: policy_state_dict[i] for i, name in enumerate(self._policy_names)}

    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        assert len(policy_state_dict) == self.num_policies

        for worker, indexes in zip(self._workers, self._worker_indexes):
            cur_dict = {i: policy_state_dict[self._policy_names[i]] for i in indexes}
            worker.set_policy_state_dict(cur_dict)
