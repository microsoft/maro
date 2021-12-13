from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from maro.rl_v3.model import MultiQNet
from maro.rl_v3.policy import DiscretePolicyGradient, RLPolicy
from maro.rl_v3.policy_trainer import MultiTrainer
from maro.rl_v3.policy_trainer.train_worker import MultiTrainWorker
from maro.rl_v3.replay_memory import RandomMultiReplayMemory
from maro.rl_v3.utils import MultiTransitionBatch, ndarray_to_tensor
from maro.utils import clone


class DiscreteMADDPGWorker(MultiTrainWorker):
    def __init__(
        self,
        name: str,
        device: torch.device,
        reward_discount: float,
        get_q_critic_net_func: Callable[[], MultiQNet],
        shared_critic: bool = False,
        critic_loss_coef: float = 1.0,
        soft_update_coef: float = 0.5,
        update_target_every: int = 5,
        q_value_loss_func: Callable = None,
        enable_data_parallelism: bool = False
    ) -> None:
        super(DiscreteMADDPGWorker, self).__init__(name, device, enable_data_parallelism)

        # Actor
        self._target_policies: Dict[int, DiscretePolicyGradient] = {}

        # Critic
        self._get_q_critic_net_func = get_q_critic_net_func
        self._q_critic_nets: Dict[int, MultiQNet] = {}
        self._target_q_critic_nets: Dict[int, MultiQNet] = {}

        #
        self._shared_critic = shared_critic

        self._reward_discount = reward_discount
        self._critic_loss_coef = critic_loss_coef
        self._q_value_loss_func = q_value_loss_func
        self._update_target_every = update_target_every
        self._soft_update_coef = soft_update_coef

    def _register_policies_impl(self, policy_dict: Dict[int, RLPolicy]) -> None:
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

    def get_worker_state_dict(self, scope: str = "all") -> dict:
        ret_dict = {}

        if scope in ("all", "actor"):
            ret_dict["policy_state"] = {i: self._policies[i].get_policy_state() for i in self._indexes}
            ret_dict["target_policy_state"] = {i: self._target_policies[i].get_policy_state() for i in self._indexes}
        if scope in ("all", "critic"):
            indexes = [0] if self._shared_critic else self._indexes
            ret_dict["critic_state"] = {i: self._q_critic_nets[i].get_net_state() for i in indexes}
            ret_dict["target_critic_state"] = {i: self._target_q_critic_nets[i].get_net_state() for i in indexes}

        return ret_dict

    def set_worker_state_dict(self, worker_state_dict: dict, scope: str = "all") -> None:
        if scope in ("all", "actor"):
            for i in self._indexes:
                self._policies[i].set_policy_state(worker_state_dict["policy_state"][i])
                self._target_policies[i].set_policy_state(worker_state_dict["target_policy_state"][i])
        if scope in ("all", "critic"):
            indexes = [0] if self._shared_critic else self._indexes
            for i in indexes:
                self._q_critic_nets[i].set_net_state(worker_state_dict["critic_state"][i])
                self._target_q_critic_nets[i].set_net_state(worker_state_dict["target_critic_state"][i])

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
        indexes = [0] if self._shared_critic else self._indexes
        for i in indexes:
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
            for i in indexes
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

    def _remote_learn(
        self,
        batch: MultiTransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
        assert self._task_queue_client is not None
        worker_id_list = self._task_queue_client.request_workers()
        # TODO: merge tensor_dict into batch_list
        batch_list = self._dispatch_batch(batch, len(worker_id_list))
        trainer_state = self.get_trainer_state_dict()
        trainer_name = self.name
        loss_info_by_name = self._task_queue_client.sumbit(
            worker_id_list, batch_list, trainer_state, trainer_name, scope)
        return loss_info_by_name[trainer_name]

    def _dispatch_batch(self, batch: MultiTransitionBatch, num_workers: int) -> List[MultiTransitionBatch]:
        batch_size = batch.states.shape[0]
        assert batch_size >= num_workers, \
            f"Batch size should be greater than or equal to num_workers, but got {batch_size} and {num_workers}."
        sub_batch_indexes = [range(batch_size)[i::num_workers] for i in range(num_workers)]
        sub_batches = [MultiTransitionBatch(
            policy_names=[],
            states=batch.states[indexes],
            actions=[action[indexes] for action in batch.actions],
            rewards=[reward[indexes] for reward in batch.rewards],
            terminals=batch.terminals[indexes],
            next_states=batch.next_states[indexes],
            agent_states=[state[indexes] for state in batch.agent_states],
            next_agent_states=[state[indexes] for state in batch.next_agent_states]
        ) for indexes in sub_batch_indexes]
        return sub_batches

    def update_critics(self, next_actions: List[torch.Tensor]) -> None:
        grads = self._get_batch_grad(
            self._batch,
            tensor_dict={"next_actions": next_actions},
            scope="critic"
        )

        for i, grad in grads["critic_grads"].items():
            self._q_critic_nets[i].train()
            self._q_critic_nets[i].apply_gradients(grad)

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

    def soft_update_target(self) -> None:
        for i in self._indexes:
            self._target_policies[i].soft_update(self._policies[i], self._soft_update_coef)

        indexes = [0] if self._shared_critic else self._indexes
        for i in indexes:
            self._target_q_critic_nets[i].soft_update(self._q_critic_nets[i], self._soft_update_coef)


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
        enable_data_parallelism: bool = False
    ) -> None:
        super(DistributedDiscreteMADDPG, self).__init__(
            name=name,
            device=device,
            enable_data_parallelism=enable_data_parallelism,
            train_batch_size=train_batch_size
        )

        self._get_q_critic_net_func = get_q_critic_net_func
        self._critic_worker: Optional[DiscreteMADDPGWorker] = None
        self._group_size = group_size
        self._replay_memory_capacity = replay_memory_capacity
        self._target_policies: List[DiscretePolicyGradient] = []
        self._shared_critic = shared_critic
        if policies is not None:
            self.register_policies(policies)

        self._num_epoch = num_epoch
        self._update_target_every = update_target_every
        self._policy_version = self._target_policy_version = 0
        self._soft_update_coef = soft_update_coef
        self._reward_discount = reward_discount
        self._critic_loss_coef = critic_loss_coef

        self._q_value_loss_func = q_value_loss_cls() if q_value_loss_cls is not None else torch.nn.MSELoss()

    def _register_policies_impl(self, policies: List[RLPolicy]) -> None:
        if self._shared_critic:
            self._critic_worker = DiscreteMADDPGWorker(
                name="critic_worker",
                reward_discount=self._reward_discount, get_q_critic_net_func=self._get_q_critic_net_func,
                shared_critic=self._shared_critic, device=self._device,
                enable_data_parallelism=self._enable_data_parallelism,
                critic_loss_coef=self._critic_loss_coef, soft_update_coef=self._soft_update_coef,
                update_target_every=self._update_target_every, q_value_loss_func=self._q_value_loss_func
            )
            self._critic_worker.register_policies({})  # Register with empty policy dict to init the critic net

        self._workers: List[DiscreteMADDPGWorker] = []
        self._worker_indexes: List[List[int]] = []
        cursor = 0
        while cursor < self.num_policies:
            cursor_end = min(cursor + self._group_size, self.num_policies)
            indexes = list(range(cursor, cursor_end))

            worker = DiscreteMADDPGWorker(
                name=f"actor_worker__{cursor}_{cursor_end - 1}",
                reward_discount=self._reward_discount, get_q_critic_net_func=self._get_q_critic_net_func,
                shared_critic=self._shared_critic, device=self._device,
                enable_data_parallelism=self._enable_data_parallelism,
                critic_loss_coef=self._critic_loss_coef, soft_update_coef=self._soft_update_coef,
                update_target_every=self._update_target_every, q_value_loss_func=self._q_value_loss_func
            )
            worker.register_policies({i: policies[i] for i in indexes})

            cursor = cursor_end
            self._workers.append(worker)
            self._worker_indexes.append(indexes)

        # Replay
        self._replay_memory = RandomMultiReplayMemory(
            capacity=self._replay_memory_capacity,
            state_dim=self._get_q_critic_net_func().state_dim,
            action_dims=[policy.action_dim for policy in policies],
            agent_states_dims=[policy.state_dim for policy in policies]
        )

    def train_step(self) -> None:
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
            self._critic_worker.set_batch(batch)
            self._critic_worker.update_critics(next_actions=next_actions)
            critic_state_dict = self._critic_worker.get_worker_state_dict(scope="critic")

            # Sync latest critic to workers
            for worker in self._workers:
                worker.set_worker_state_dict(critic_state_dict, scope="critic")
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
        self._try_soft_update_target()

    def _try_soft_update_target(self) -> None:
        self._policy_version += 1
        if self._policy_version - self._target_policy_version == self._update_target_every:
            if self._shared_critic:
                self._critic_worker.soft_update_target()

            for worker in self._workers:
                worker.soft_update_target()

            self._target_policy_version = self._policy_version

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
