from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from maro.rl_v3.model import MultiQNet
from maro.rl_v3.policy import DiscretePolicyGradient, RLPolicy
from maro.rl_v3.replay_memory import RandomMultiReplayMemory
from maro.rl_v3.utils import MultiTransitionBatch, ndarray_to_tensor
from maro.utils import clone

from .abs_trainer import MultiTrainer


class DiscreteMADDPG(MultiTrainer):
    def __init__(
        self,
        name: str,
        reward_discount: float,
        get_q_critic_net_func: Callable[[], MultiQNet],
        policies: List[RLPolicy] = None,
        replay_memory_capacity: int = 10000,
        num_epoch: int = 10,
        update_target_every: int = 5,
        soft_update_coef: float = 0.5,
        train_batch_size: int = 32,
        q_value_loss_cls: Callable = None,
        device: str = None,
        critic_loss_coef: float = 1.0,
        shared_critic: bool = False

    ) -> None:
        super(DiscreteMADDPG, self).__init__(name, device)

        self._get_q_critic_net_func = get_q_critic_net_func
        self._q_critic_nets: Optional[List[MultiQNet]] = None
        self._target_q_critic_nets: Optional[List[MultiQNet]] = None
        self._replay_memory_capacity = replay_memory_capacity
        self._target_policies: Optional[List[DiscretePolicyGradient]] = None
        if policies is not None:
            self.register_policies(policies)

        self._num_epoch = num_epoch
        self._update_target_every = update_target_every
        self._policy_version = self._target_policy_version = 0
        self._soft_update_coef = soft_update_coef
        self._train_batch_size = train_batch_size
        self._reward_discount = reward_discount
        self._critic_loss_coef = critic_loss_coef
        self._shared_critic = shared_critic

        self._q_value_loss_func = q_value_loss_cls() if q_value_loss_cls is not None else torch.nn.MSELoss()

    def _record_impl(self, transition_batch: MultiTransitionBatch) -> None:
        self._replay_memory.put(transition_batch)

    def _register_policies_impl(self, policies: List[RLPolicy]) -> None:
        assert all(isinstance(policy, DiscretePolicyGradient) for policy in policies)

        self._policies = policies
        self._policy_dict = {
            policy.name: policy for policy in policies
        }
        if self._shared_critic:
            q_critic_net = self._get_q_critic_net_func()
            q_critic_net.to(self._device)
            self._q_critic_nets = [q_critic_net for _ in range(self.num_policies)]
        else:
            self._q_critic_nets = [self._get_q_critic_net_func().to(self._device) for _ in range(self.num_policies)]

        self._replay_memory = RandomMultiReplayMemory(
            capacity=self._replay_memory_capacity,
            state_dim=self._q_critic_nets[0].state_dim,
            action_dims=[policy.action_dim for policy in policies],
            agent_states_dims=[policy.state_dim for policy in policies]
        )

        self._target_policies: List[DiscretePolicyGradient] = []
        for policy in self._policies:
            target_policy = clone(policy)
            target_policy.set_name(f"target_{policy.name}")
            self._target_policies.append(target_policy)

        for policy in self._target_policies:
            policy.eval()

        self._target_q_critic_nets = [clone(net) for net in self._q_critic_nets]
        for i in range(self.num_policies):
            self._target_q_critic_nets[i].eval()
            self._target_q_critic_nets[i].to(self._device)

        for policy in self._target_policies:
            policy.to_device(self._device)

    def _get_batch(self, batch_size: int = None) -> MultiTransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def _train_step_impl(self, data_parallel: bool = False) -> None:
        for _ in range(self._num_epoch):
            self._improve(self._get_batch(), data_parallel)

    def _improve(self, batch: MultiTransitionBatch, data_parallel: bool) -> None:
        grads = self._get_batch_grad(batch, scope="critic", data_parallel=data_parallel)
        for net, grad in zip(self._q_critic_nets, grads["critic_grads"]):
            net.train()
            net.apply_gradients(grad)
            if self._shared_critic:
                break

        grads = self._get_batch_grad(batch, scope="actor", data_parallel=data_parallel)
        for policy, grad in zip(self._policies, grads["actor_grads"]):
            policy.train()
            policy.apply_gradients(grad)

        self._update_target_policy()

    def _batch_grad_worker(
        self,
        batch: MultiTransitionBatch,
        scope: str = "all"
    ) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        assert scope in ("all", "actor", "critic"), \
            f"Unrecognized scope {scope}. Excepting 'all', 'actor', or 'critic'."

        states = ndarray_to_tensor(batch.states, self._device)  # x
        agent_states = [ndarray_to_tensor(agent_state, self._device) for agent_state in batch.agent_states]  # o
        actions = [ndarray_to_tensor(action, self._device) for action in batch.actions]  # a

        grad_dict = {}
        if scope in ("all", "critic"):
            with torch.no_grad():
                next_actions = [
                    policy.get_actions_tensor(agent_state)  # a' = miu'(o)
                    for policy, agent_state in zip(self._target_policies, agent_states)
                ]

            next_states = ndarray_to_tensor(batch.next_states, self._device)  # x'
            rewards = ndarray_to_tensor(np.vstack([reward for reward in batch.rewards]), self._device)  # r
            terminals = ndarray_to_tensor(batch.terminals, self._device)  # d

            for net in self._q_critic_nets:
                net.train()

            critic_losses = []
            if self._shared_critic:
                with torch.no_grad():
                    next_q_values = self._target_q_critic_nets[0].q_values(
                        states=next_states,  # x'
                        actions=next_actions  # a'
                    )  # Q'(x', a')
                # sum(rewards) for shard critic
                target_q_values = (rewards.sum(0) + self._reward_discount * (1 - terminals.float()) * next_q_values)
                q_values = self._q_critic_nets[0].q_values(
                    states=states,  # x
                    actions=actions  # a
                )  # Q(x, a)
                critic_loss = self._q_value_loss_func(q_values, target_q_values.detach()) * self._critic_loss_coef
                critic_losses.append(critic_loss)
            else:
                for i in range(self.num_policies):
                    with torch.no_grad():
                        next_q_values = self._target_q_critic_nets[i].q_values(
                            states=next_states,  # x'
                            actions=next_actions)  # a'
                    target_q_values = (
                        rewards[i] + self._reward_discount * (1 - terminals.float()) * next_q_values)
                    q_values = self._q_critic_nets[i].q_values(
                        states=states,  # x
                        actions=actions  # a
                    )  # Q(x, a)
                    critic_loss = self._q_value_loss_func(q_values, target_q_values.detach()) * self._critic_loss_coef
                    critic_losses.append(critic_loss)

            grad_dict["critic_grads"] = [
                net.get_gradients(loss)
                for net, loss in zip(self._q_critic_nets, critic_losses)
            ]

        if scope in ("all", "actor"):
            for policy in self._policies:
                policy.train()

            latest_actions = []
            latest_action_logps = []
            for policy, agent_state in zip(self._policies, agent_states):
                assert isinstance(policy, DiscretePolicyGradient)
                latest_actions.append(policy.get_actions_tensor(agent_state))  # a = miu(o)
                latest_action_logps.append(policy.get_state_action_logps(
                    agent_state,  # o
                    latest_actions[-1]  # a
                ))  # log pi(a|o)

            actor_losses = []
            for i in range(self.num_policies):
                # Update actor
                self._q_critic_nets[i].freeze()

                action_backup = actions[i]
                actions[i] = latest_actions[i]  # Replace latest action
                actor_loss = -(self._q_critic_nets[i].q_values(
                    states=states,  # x
                    actions=actions  # [a^j_1, ..., a_i, ..., a^j_N]
                ) * latest_action_logps[i]).mean()  # Q(x, a^j_1, ..., a_i, ..., a^j_N)
                actor_losses.append(actor_loss)

                actions[i] = action_backup  # Restore original action
                self._q_critic_nets[i].unfreeze()

            grad_dict["actor_grads"] = [
                policy.get_gradients(loss)
                for policy, loss in zip(self._policies, actor_losses)
            ]

        return grad_dict

    def _update_target_policy(self) -> None:
        self._policy_version += 1
        if self._policy_version - self._target_policy_version == self._update_target_every:
            for policy, target_policy in zip(self._policies, self._target_policies):
                target_policy.soft_update(policy, self._soft_update_coef)
            for critic, target_critic in zip(self._q_critic_nets, self._target_q_critic_nets):
                target_critic.soft_update(critic, self._soft_update_coef)
                if self._shared_critic:
                    break  # only update once for shared critic
            self._target_policy_version = self._policy_version

    def get_trainer_state_dict(self) -> dict:
        return {
            "policy_state": self.get_policy_state_dict(),
            "target_policy_state": [policy.get_policy_state() for policy in self._target_policies],
            "critic_state": [net.get_net_state() for net in self._q_critic_nets],
            "target_critic_state": [net.get_net_state() for net in self._target_q_critic_nets]
        }

    def set_trainer_state_dict(self, trainer_state_dict: dict) -> None:
        self.set_policy_state_dict(trainer_state_dict["policy_state"])
        for policy, policy_state in zip(self._target_policies, trainer_state_dict["target_policy_state"]):
            policy.set_policy_state(policy_state)
        for net, net_state in zip(self._q_critic_nets, trainer_state_dict["critic_state"]):
            net.set_net_state(net_state)
        for net, net_state in zip(self._target_q_critic_nets, trainer_state_dict["target_critic_state"]):
            net.set_net_state(net_state)
