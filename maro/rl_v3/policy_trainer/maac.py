from typing import Callable, List, Optional

import torch

from maro.rl_v3.model.multi_q_net import MultiQNet
from maro.rl_v3.policy import DiscretePolicyGradient, RLPolicy
from maro.rl_v3.policy_trainer import RandomMultiReplayMemory
from maro.rl_v3.policy_trainer.abs_trainer import MultiTrainer
from maro.rl_v3.utils import MultiTransitionBatch
from maro.utils import clone


class DiscreteMultiActorCritic(MultiTrainer):
    def __init__(
        self,
        name: str,
        state_dim: int,
        reward_discount: float,
        get_v_critic_net_func: Callable[[], MultiQNet],
        policies: List[RLPolicy] = None,
        replay_memory_capacity: int = 10000,
        num_epoch: int = 10,
        update_target_every: int = 5,
        soft_update_coef: float = 1.0,
        train_batch_size: int = 32,
        q_value_loss_cls: Callable = None

    ) -> None:
        super(DiscreteMultiActorCritic, self).__init__(name)

        self._get_v_critic_net_func = get_v_critic_net_func
        self._q_critic_net: Optional[MultiQNet] = None
        self._target_q_critic_net: Optional[MultiQNet] = None
        self._replay_memory_capacity = replay_memory_capacity
        self._state_dim = state_dim
        self._target_policies: Optional[List[DiscretePolicyGradient]] = None
        if policies is not None:
            self.register_policies(policies)

        self._num_epoch = num_epoch
        self._update_target_every = update_target_every
        self._policy_ver = self._target_policy_ver = 0
        self._soft_update_coef = soft_update_coef
        self._train_batch_size = train_batch_size
        self._reward_discount = reward_discount

        self._q_value_loss_func = q_value_loss_cls() if q_value_loss_cls is not None else torch.nn.MSELoss()

    def _record_impl(self, transition_batch: MultiTransitionBatch) -> None:
        self._replay_memory.put(transition_batch)

    def register_policies(self, policies: List[RLPolicy]) -> None:
        assert all(isinstance(policy, DiscretePolicyGradient) for policy in policies)

        self._policies = policies
        self._policy_dict = {
            policy.name: policy for policy in policies
        }
        self._q_critic_net = self._get_v_critic_net_func()
        self._replay_memory = RandomMultiReplayMemory(
            capacity=self._replay_memory_capacity, state_dim=self._state_dim,
            action_dims=[policy.action_dim for policy in policies]
        )

        self._target_policies: List[DiscretePolicyGradient] = [clone(policy) for policy in self._policies]
        for policy in self._target_policies:
            policy.eval()
        self._target_q_critic_net: MultiQNet = clone(self._q_critic_net)
        self._target_q_critic_net.eval()

    def _get_batch(self, batch_size: int = None) -> MultiTransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def train_step(self) -> None:
        for _ in range(self._num_epoch):
            self._improve(self._get_batch())
            self._update_target_policy()

    def _improve(self, batch: MultiTransitionBatch) -> None:
        """
        https://arxiv.org/pdf/1706.02275.pdf
        """
        for policy in self._policies:
            policy.train()

        states = self.ndarray_to_tensor(batch.states)  # x
        next_states = self.ndarray_to_tensor(batch.next_states)  # x'
        agent_states = [self.ndarray_to_tensor(agent_state) for agent_state in batch.agent_states]  # o
        actions = [self.ndarray_to_tensor(action) for action in batch.actions]  # a
        rewards = [self.ndarray_to_tensor(reward) for reward in batch.rewards]  # r
        terminals = self.ndarray_to_tensor(batch.terminals)  # d

        with torch.no_grad():
            next_actions = [
                policy.get_actions_tensor(agent_state)  # a' = miu'(o)
                for policy, agent_state in zip(self._target_policies, agent_states)
            ]
            next_q_values = self._target_q_critic_net.q_values(
                states=next_states,  # x'
                actions=next_actions  # a'
            )  # Q'(x', a')

        critic_losses = []
        actor_losses = []
        for i in range(len(self._policies)):
            # y = r + gamma * (1 - d) * Q'
            target_q_values = (rewards[i] + self._reward_discount * (1 - terminals) * next_q_values).detach()
            q_values = self._q_critic_net.q_values(
                states=states,  # x
                actions=actions  # a
            )  # Q(x, a)

            # Replace action for current agent
            action_backup = actions[i]
            actions[i] = self._policies[i].get_actions_tensor(agent_states[i])  # a = miu(o)

            q_loss = self._q_value_loss_func(q_values, target_q_values)  # MSE(Q(x, a), Q'(x', a'))
            policy_loss = -self._q_critic_net.q_values(
                states=states,  # x
                actions=actions  # [a^j_1, ..., a_i, ..., a^j_N]
            )  # Q(x, a^j_1, ..., a_i, ..., a^j_N)

            actor_losses.append(policy_loss)
            critic_losses.append(q_loss)

            # Restore old action
            actions[i] = action_backup

        # Update Q first, then freeze Q and update miu.
        for q_loss in critic_losses:
            self._q_critic_net.step(q_loss * 0.1)  # TODO
        self._q_critic_net.freeze()
        for policy, actor_loss in zip(self._policies, actor_losses):
            policy.step(actor_loss)
        self._q_critic_net.unfreeze()

    def _update_target_policy(self) -> None:
        self._policy_ver += 1
        if self._policy_ver - self._target_policy_ver == self._update_target_every:
            for policy, target_policy in zip(self._policies, self._target_policies):
                target_policy.soft_update(policy, self._soft_update_coef)
            self._target_q_critic_net.soft_update(self._q_critic_net, self._soft_update_coef)
            self._target_policy_ver = self._policy_ver
