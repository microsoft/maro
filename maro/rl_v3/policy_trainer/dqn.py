from typing import Dict, Optional

import torch

from maro.rl_v3.policy import ValueBasedPolicy
from maro.rl_v3.replay_memory import RandomReplayMemory
from maro.rl_v3.utils import TransitionBatch, ndarray_to_tensor
from maro.utils import clone

from .abs_trainer import SingleTrainer


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
        random_overwrite: bool = False,
        device: str = None
    ) -> None:
        super(DQN, self).__init__(name, device)

        self._policy: Optional[ValueBasedPolicy] = None
        self._target_policy: Optional[ValueBasedPolicy] = None
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
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def _train_step_impl(self) -> None:
        for _ in range(self._num_epochs):
            self._improve(self._get_batch())

        self._policy_ver += 1
        if self._policy_ver - self._target_policy_ver == self._update_target_every:
            self._target_policy.soft_update(self._policy, self._soft_update_coef)
            self._target_policy_ver = self._policy_ver

    def _get_loss(self, batch: TransitionBatch) -> torch.Tensor:
        self._policy.train()
        states = ndarray_to_tensor(batch.states, self._device)
        next_states = ndarray_to_tensor(batch.next_states, self._device)
        actions = ndarray_to_tensor(batch.actions, self._device)
        rewards = ndarray_to_tensor(batch.rewards, self._device)
        terminals = ndarray_to_tensor(batch.terminals, self._device).float()

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
        return self._loss_func(q_values, target_q_values)

    def atomic_get_batch_grad(self, batch: TransitionBatch, scope: str = "all") -> Dict[str, Dict[str, torch.Tensor]]:
        assert scope == "all", f"Unrecognized scope {scope}. Excepting 'all'."

        self._policy.train()
        states = ndarray_to_tensor(batch.states, self._device)
        next_states = ndarray_to_tensor(batch.next_states, self._device)
        actions = ndarray_to_tensor(batch.actions, self._device)
        rewards = ndarray_to_tensor(batch.rewards, self._device)
        terminals = ndarray_to_tensor(batch.terminals, self._device).float()

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
        loss: torch.Tensor = self._loss_func(q_values, target_q_values)

        return {"grad": self._policy.get_gradients(loss)}

    def _improve(self, batch: TransitionBatch) -> None:
        grad_dict = self._get_batch_grad(batch)

        self._policy.train()
        self._policy.apply_gradients(grad_dict["grad"])

    def _register_policy_impl(self, policy: ValueBasedPolicy) -> None:
        assert isinstance(policy, ValueBasedPolicy)

        self._policy = policy
        self._target_policy: ValueBasedPolicy = clone(policy)
        self._target_policy.set_name(f"target_{policy.name}")
        self._target_policy.eval()
        self._target_policy.to_device(self._device)

        self._replay_memory = RandomReplayMemory(
            capacity=self._replay_memory_capacity, state_dim=policy.state_dim,
            action_dim=policy.action_dim, random_overwrite=self._random_overwrite
        )

    def get_trainer_state_dict(self) -> dict:
        return {
            "policy_state": self.get_policy_state_dict(),
            "target_policy_state": self._target_policy.get_policy_state()
        }

    def set_trainer_state_dict(self, trainer_state_dict: dict) -> None:
        self.set_policy_state_dict(trainer_state_dict["policy_state"])
        self._target_policy.set_policy_state(trainer_state_dict["target_policy_state"])
