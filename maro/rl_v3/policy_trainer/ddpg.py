from typing import Callable, Optional

import torch

from maro.rl_v3.model import QNet
from maro.rl_v3.policy import ContinuousRLPolicy
from maro.rl_v3.policy_trainer import RandomReplayMemory, SingleTrainer
from maro.rl_v3.utils import TransitionBatch
from maro.utils import clone


class DDPG(SingleTrainer):
    def __init__(
        self,
        name: str,
        get_q_critic_net_func: Callable[[], QNet],
        reward_discount: float,
        q_value_loss_cls: Callable = None,
        policy: ContinuousRLPolicy = None,
        random_overwrite: bool = False,
        replay_memory_capacity: int = 1000000,
        num_epochs: int = 1,
        update_target_every: int = 5,
        soft_update_coef: float = 1.0,
        train_batch_size: int = 32
    ) -> None:
        super(DDPG, self).__init__(name=name)

        self._policy: ContinuousRLPolicy = Optional[ContinuousRLPolicy]
        self._target_policy: ContinuousRLPolicy = Optional[ContinuousRLPolicy]
        self._q_critic_net: QNet = Optional[QNet]
        self._target_q_critic_net: QNet = Optional[QNet]
        self._get_q_critic_net_func = get_q_critic_net_func
        self._replay_memory_capacity = replay_memory_capacity
        self._replay_memory = Optional[RandomReplayMemory]
        self._random_overwrite = random_overwrite
        if policy is not None:
            self.register_policy(policy)

        self._num_epochs = num_epochs
        self._policy_ver = self._target_policy_ver = 0
        self._update_target_every = update_target_every
        self._soft_update_coef = soft_update_coef
        self._train_batch_size = train_batch_size
        self._reward_discount = reward_discount
        self._q_value_loss_func = q_value_loss_cls() if q_value_loss_cls is not None else torch.nn.MSELoss()

    def _record_impl(self, policy_name: str, transition_batch: TransitionBatch) -> None:
        self._replay_memory.put(transition_batch)

    def register_policy(self, policy: ContinuousRLPolicy) -> None:
        assert isinstance(policy, ContinuousRLPolicy)
        self._policy = policy
        self._target_policy = clone(self._policy)
        self._target_policy.eval()
        self._replay_memory = RandomReplayMemory(
            capacity=self._replay_memory_capacity, state_dim=policy.state_dim,
            action_dim=policy.action_dim, random_overwrite=self._random_overwrite
        )
        self._q_critic_net = self._get_q_critic_net_func()
        self._target_q_critic_net: QNet = clone(self._q_critic_net)
        self._target_q_critic_net.eval()

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def train_step(self) -> None:
        for _ in range(self._num_epochs):
            self.improve(self._get_batch())
        self._policy_ver += 1
        if self._policy_ver - self._target_policy_ver == self._update_target_every:
            self._target_policy.soft_update(self._policy, self._soft_update_coef)
            self._target_q_critic_net.soft_update(self._q_critic_net, self._soft_update_coef)
            self._target_policy_ver = self._policy_ver

    def improve(self, batch: TransitionBatch) -> None:
        """
        Reference: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        """
        self._policy.train()

        states = self._policy.ndarray_to_tensor(batch.states)  # s
        next_states = self._policy.ndarray_to_tensor(batch.next_states)  # s'
        actions = self._policy.ndarray_to_tensor(batch.actions)  # a
        rewards = self._policy.ndarray_to_tensor(batch.rewards)  # r
        terminals = self._policy.ndarray_to_tensor(batch.terminals)  # d

        with torch.no_grad():
            next_q_values = self._target_q_critic_net.q_values(
                states=next_states,  # s'
                actions=self._target_policy.get_actions_tensor(next_states)  # miu_targ(s')
            )  # Q_targ(s', miu_targ(s'))

        # y(r, s', d) = r + gamma * (1 - d) * Q_targ(s', miu_targ(s'))
        target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()

        q_values = self._q_critic_net.q_values(states=states, actions=actions)  # Q(s, a)
        q_loss = self._q_value_loss_func(q_values, target_q_values)  # MSE(Q(s, a), y(r, s', d))
        policy_loss = -self._q_critic_net.q_values(
            states=states,  # s
            actions=self._policy.get_actions_tensor(states)  # miu(s)
        ).mean()  # -Q(s, miu(s))

        # Update
        self._policy.step(policy_loss)
        self._q_critic_net.step(q_loss * 0.1)  # TODO
