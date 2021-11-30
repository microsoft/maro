from typing import Callable, Dict, List, Optional

import torch

from maro.rl_v3.model import MultiQNet
from maro.rl_v3.policy import DiscretePolicyGradient, RLPolicy
from maro.rl_v3.replay_memory import RandomMultiReplayMemory
from maro.rl_v3.utils import MultiTransitionBatch, ndarray_to_tensor
from maro.utils import clone
from .abs_trainer import MultiTrainer


class MADDPG(MultiTrainer):
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
        super(MADDPG, self).__init__(name, device)

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
            self._q_critic_nets = [q_critic_net for i in range(self.num_policies)]
        else:
            self._q_critic_nets = [self._get_q_critic_net_func().to(self._device) for i in range(self.num_policies)]

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

    def train_step(self) -> None:
        for _ in range(self._num_epoch):
            train_batch = self._get_batch()
            # iteratively update critic & actors
            loss = self.get_batch_loss(train_batch, scope="critic")
            for critic_net, critic_loss in zip(self._q_critic_nets, loss["critic_losses"]):
                critic_net.step(critic_loss)
                if self._shared_critic:
                    break  # only update once for shared critic

            loss = self.get_batch_loss(train_batch, scope="actor")
            for policy, actor_loss in zip(self._policies, loss["actor_losses"]):
                policy.step(actor_loss)

            self._update_target_policy()

    def get_batch_loss(self, batch: MultiTransitionBatch, scope="all") -> Dict[str, torch.Tensor]:
        """Get loss with a batch of data. If scope is specified, return the expected loss of scope.

        Args:
            batch (MultiTransitionBatch): The batch of multi-agent experience data.
            scope (str): The expected scope to compute loss. Should be in ['all', 'critic', 'actor'].

        Returns:
            loss_info (Dict[str, torch.Tensor]): Loss of each scope.
        """
        assert scope in ["all", "critic", "actor"], f'scope should in ["all", "critic", "actor"] but get {scope}.'
        loss_info = dict()
        if scope == "all" or scope == "critic":
            critic_losses = self._get_critic_losses(batch)
            loss_info["critic_losses"] = critic_losses
        if scope == "all" or scope == "actor":
            actor_losses = self._get_actor_losses(batch)
            loss_info["actor_losses"] = actor_losses
        return loss_info

    def _get_critic_losses(self, batch: MultiTransitionBatch) -> List[torch.Tensor]:
        states = ndarray_to_tensor(batch.states, self._device)  # x
        next_states = ndarray_to_tensor(batch.next_states, self._device)  # x'
        agent_states = [ndarray_to_tensor(agent_state, self._device) for agent_state in batch.agent_states]  # o
        actions = [ndarray_to_tensor(action, self._device) for action in batch.actions]  # a
        rewards = [ndarray_to_tensor(reward, self._device) for reward in batch.rewards]  # r
        terminals = ndarray_to_tensor(batch.terminals, self._device)  # d

        with torch.no_grad():
            next_actions = [
                policy.get_actions_tensor(agent_state)  # a' = miu'(o)
                for policy, agent_state in zip(self._target_policies, agent_states)
            ]

        critic_losses = []
        if self._shared_critic:
            with torch.no_grad():
                next_q_values = self._target_q_critic_nets[0].q_values(
                    states=next_states,  # x'
                    actions=next_actions  # a'
                )  # Q'(x', a')
            # sum(rewards) for shard critic
            target_q_values = (sum(rewards) + self._reward_discount * (1 - terminals.float()) * next_q_values).detach()
            q_values = self._q_critic_nets[0].q_values(
                states=states,  # x
                actions=actions  # a
            )  # Q(x, a)
            critic_loss = self._q_value_loss_func(q_values, target_q_values) * self._critic_loss_coef
            critic_losses.append(critic_loss)
        else:
            for i in range(self.num_policies):
                with torch.no_grad():
                    next_q_values = self._target_q_critic_nets[i].q_values(
                        states=next_states,  # x'
                        actions=next_actions)  # a'
                target_q_values = (
                    rewards[i] + self._reward_discount * (1 - terminals.float()) * next_q_values).detach()
                q_values = self._q_critic_nets[i].q_values(
                    states=states,  # x
                    actions=actions  # a
                )  # Q(x, a)
                critic_loss = self._q_value_loss_func(q_values, target_q_values) * self._critic_loss_coef
                critic_losses.append(critic_loss)

        return critic_losses

    def _get_actor_losses(self, batch: MultiTransitionBatch) -> List[torch.Tensor]:
        for policy in self._policies:
            policy.train()

        states = ndarray_to_tensor(batch.states, self._device)  # x
        agent_states = [ndarray_to_tensor(agent_state, self._device) for agent_state in batch.agent_states]  # o
        actions = [ndarray_to_tensor(action, self._device) for action in batch.actions]  # a

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
        for i in range(len(self._policies)):
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
        return actor_losses

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
