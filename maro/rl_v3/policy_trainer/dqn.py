from typing import Optional

import torch

from maro.rl_v3.policy.discrete_rl_policy import ValueBasedPolicy
from maro.rl_v3.policy_trainer.abs_trainer import SingleTrainer
from maro.rl_v3.policy_trainer.replay_memory import RandomReplayMemory
from maro.rl_v3.utils.transition_batch import TransitionBatch
from maro.utils import clone


class DQN(SingleTrainer):
    def __init__(
        self,
        name: str,
        policy: ValueBasedPolicy = None,
        replay_memory_capacity: int = 100000,
        train_batch_size: int = 128,
        num_epochs: int = 1,
        reward_discount: float = 0.9,
        update_target_every: int = 5,
        soft_update_coef: float = 0.1,
        double: bool = False
    ) -> None:
        super(DQN, self).__init__(name)

        self._policy: Optional[ValueBasedPolicy] = None
        self._target_policy: Optional[ValueBasedPolicy] = None
        if policy is not None:
            self.register_policy(policy)

        self._replay_memory = RandomReplayMemory(
            capacity=replay_memory_capacity, state_dim=policy.state_dim,
            action_dim=policy.action_dim, random_overwrite=True
        )

        self._train_batch_size = train_batch_size
        self._num_epochs = num_epochs
        self._reward_discount = reward_discount

        self._policy_ver = self._target_policy_ver = 0
        self._update_target_every = update_target_every
        self._soft_update_coef = soft_update_coef
        self._double = double

    def _record_impl(self, policy_name: str, transition_batch: TransitionBatch) -> None:
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def train_step(self) -> None:
        for _ in range(self._num_epochs):
            self.improve(self._get_batch())

    def improve(self, batch: TransitionBatch) -> None:
        self._policy.train()
        states = self._policy.ndarray_to_tensor(batch.states)
        next_states = self._policy.ndarray_to_tensor(batch.next_states)
        actions = self._policy.ndarray_to_tensor(batch.actions)
        rewards = self._policy.ndarray_to_tensor(batch.rewards)
        terminals = self._policy.ndarray_to_tensor(batch.terminals)

        with torch.no_grad():
            if self._double:
                actions_by_eval_policy = self._policy.get_actions_tensor(states)
                next_q_values = self._target_policy.q_values_tensor(next_states, actions_by_eval_policy)
            else:
                actions = self._target_policy.get_actions_tensor(next_states)
                next_q_values = self._target_policy.q_values_tensor(next_states, actions)
        target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()

        q_values = self._policy.q_values_tensor(states, actions)
        # td_errors = target_q_values - q_values
        loss = torch.nn.MSELoss()(q_values, target_q_values)

        self._policy.step(loss)
        self._policy_ver += 1
        if self._policy_ver - self._target_policy_ver == self._update_target_every:
            self._target_policy.soft_update(self._policy, self._soft_update_coef)
            self._target_policy_ver = self._policy_ver

    def register_policy(self, policy: ValueBasedPolicy) -> None:
        self._policy = policy
        self._target_policy: ValueBasedPolicy = clone(policy)
        self._target_policy.eval()
