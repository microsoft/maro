from typing import Optional

import torch

from maro.rl_v3.policy import ValueBasedPolicy
from maro.rl_v3.utils import TransitionBatch
from maro.utils import clone
from .abs_trainer import SingleTrainer
from .replay_memory import RandomReplayMemory


class DQN(SingleTrainer):
    """
    TODO: docs.
    """
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
        double: bool = False,
        random_overwrite: bool = False
    ) -> None:
        super(DQN, self).__init__(name)

        self._policy: Optional[ValueBasedPolicy] = None
        self._target_policy: Optional[ValueBasedPolicy] = None
        self._replay_memory: Optional[RandomReplayMemory] = None  # Will be created in `register_policy`
        self._replay_memory_capacity = replay_memory_capacity
        self._random_overwrite = random_overwrite
        if policy is not None:
            self.register_policy(policy)

        self._train_batch_size = train_batch_size
        self._num_epochs = num_epochs
        self._reward_discount = reward_discount

        self._policy_ver = self._target_policy_ver = 0
        self._update_target_every = update_target_every
        self._soft_update_coef = soft_update_coef
        self._double = double

        self._loss_func = torch.nn.MSELoss()

    def _record_impl(self, policy_name: str, transition_batch: TransitionBatch) -> None:
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        from maro.utils import set_seeds
        set_seeds(987)
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def train_step(self) -> None:
        for _ in range(self._num_epochs):
            self.improve(self._get_batch())
        self._policy_ver += 1
        if self._policy_ver - self._target_policy_ver == self._update_target_every:
            self._target_policy.soft_update(self._policy, self._soft_update_coef)
            self._target_policy_ver = self._policy_ver

    def improve(self, batch: TransitionBatch) -> None:
        self._policy.train()
        states = self._policy.ndarray_to_tensor(batch.states)
        next_states = self._policy.ndarray_to_tensor(batch.next_states)
        actions = self._policy.ndarray_to_tensor(batch.actions)
        rewards = self._policy.ndarray_to_tensor(batch.rewards)
        terminals = self._policy.ndarray_to_tensor(batch.terminals).float()

        with torch.no_grad():
            if self._double:
                self._policy.exploit()
                actions_by_eval_policy = self._policy.get_actions_tensor(next_states)
                next_q_values = self._target_policy.q_values_tensor(next_states, actions_by_eval_policy)
            else:
                self._target_policy.exploit()
                actions = self._target_policy.get_actions_tensor(next_states)
                next_q_values = self._target_policy.q_values_tensor(next_states, actions)
        target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()

        q_values = self._policy.q_values_tensor(states, actions)
        # td_errors = target_q_values - q_values
        loss = self._loss_func(q_values, target_q_values)

        self._policy.step(loss)

    def register_policy(self, policy: ValueBasedPolicy) -> None:
        assert isinstance(policy, ValueBasedPolicy)

        self._policy = policy
        self._target_policy: ValueBasedPolicy = clone(policy)
        self._target_policy.eval()

        self._replay_memory = RandomReplayMemory(
            capacity=self._replay_memory_capacity, state_dim=policy.state_dim,
            action_dim=policy.action_dim, random_overwrite=self._random_overwrite
        )
