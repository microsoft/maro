from typing import Callable, Dict, Optional

import torch

from maro.rl_v3.model import QNet
from maro.rl_v3.policy import ContinuousRLPolicy
from maro.rl_v3.replay_memory import RandomReplayMemory
from maro.rl_v3.utils import TransitionBatch, ndarray_to_tensor
from maro.utils import clone

from .abs_trainer import SingleTrainer


class DDPGActorModule(object):
    def __init__(
        self
    ) -> None:
        super(DDPGActorModule, self).__init__()

        self._policy: ContinuousRLPolicy = Optional[ContinuousRLPolicy]
        self._target_policy: ContinuousRLPolicy = Optional[ContinuousRLPolicy]

    def _register_policy_impl(self, policy: ContinuousRLPolicy) -> None:
        assert isinstance(policy, ContinuousRLPolicy)
        self._policy = policy
        self._target_policy = clone(self._policy)
        self._target_policy.set_name(f"target_{policy.name}")
        self._target_policy.eval()

    def get_batch_grad(self) -> Dict[str, torch.Tensor]:





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
        train_batch_size: int = 32,
        critic_loss_coef: float = 0.1,
        device: str = None
    ) -> None:
        super(DDPG, self).__init__(name=name, device=device)

        self._policy: ContinuousRLPolicy = Optional[ContinuousRLPolicy]
        self._target_policy: ContinuousRLPolicy = Optional[ContinuousRLPolicy]
        self._q_critic_net: QNet = Optional[QNet]
        self._target_q_critic_net: QNet = Optional[QNet]
        self._get_q_critic_net_func = get_q_critic_net_func
        self._replay_memory_capacity = replay_memory_capacity
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
        self._critic_loss_coef = critic_loss_coef

    def _record_impl(self, policy_name: str, transition_batch: TransitionBatch) -> None:
        self._replay_memory.put(transition_batch)

    def _register_policy_impl(self, policy: ContinuousRLPolicy) -> None:
        assert isinstance(policy, ContinuousRLPolicy)
        self._policy = policy
        self._target_policy = clone(self._policy)
        self._target_policy.set_name(f"target_{policy.name}")
        self._target_policy.eval()
        self._replay_memory = RandomReplayMemory(
            capacity=self._replay_memory_capacity, state_dim=policy.state_dim,
            action_dim=policy.action_dim, random_overwrite=self._random_overwrite
        )
        self._q_critic_net = self._get_q_critic_net_func()
        self._target_q_critic_net: QNet = clone(self._q_critic_net)
        self._target_q_critic_net.eval()

        self._target_policy.to_device(self._device)
        self._q_critic_net.to(self._device)
        self._target_q_critic_net.to(self._device)

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def _train_step_impl(self) -> None:
        for _ in range(self._num_epochs):
            self._improve(self._get_batch())
            self._update_target_policy()

    def get_batch_grad(self, batch: TransitionBatch, scope: str = "all") -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Reference: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        """

        assert scope in ("all", "actor", "critic"), \
            f"Unrecognized scope {scope}. Excepting 'all', 'actor', or 'critic'."

        self._q_critic_net.train()
        self._policy.train()

        states = ndarray_to_tensor(batch.states, self._device)  # s

        grad_dict = {}
        if scope in ("all", "critic"):
            next_states = ndarray_to_tensor(batch.next_states, self._device)  # s'
            actions = ndarray_to_tensor(batch.actions, self._device)  # a
            rewards = ndarray_to_tensor(batch.rewards, self._device)  # r
            terminals = ndarray_to_tensor(batch.terminals, self._device)  # d

            with torch.no_grad():
                next_q_values = self._target_q_critic_net.q_values(
                    states=next_states,  # s'
                    actions=self._target_policy.get_actions_tensor(next_states)  # miu_targ(s')
                )  # Q_targ(s', miu_targ(s'))

            # y(r, s', d) = r + gamma * (1 - d) * Q_targ(s', miu_targ(s'))
            target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()

            q_values = self._q_critic_net.q_values(states=states, actions=actions)  # Q(s, a)
            critic_loss = self._q_value_loss_func(q_values, target_q_values)  # MSE(Q(s, a), y(r, s', d))

            grad_dict["critic_grad"] = self._q_critic_net.get_gradients(critic_loss * self._critic_loss_coef)

        if scope in ("all", "actor"):
            policy_loss = -self._q_critic_net.q_values(
                states=states,  # s
                actions=self._policy.get_actions_tensor(states)  # miu(s)
            ).mean()  # -Q(s, miu(s))

            grad_dict["actor_grad"] = self._policy.get_gradients(policy_loss)

        return grad_dict

    def _improve(self, batch: TransitionBatch) -> None:
        grad_dict = self._get_batch_grad(batch, scope="critic")
        self._q_critic_net.train()
        self._q_critic_net.apply_gradients(grad_dict["critic_grad"])

        grad_dict = self._get_batch_grad(batch, scope="actor")
        self._policy.train()
        self._policy.apply_gradients(grad_dict["actor_grad"])

    def _update_target_policy(self) -> None:
        self._policy_ver += 1
        if self._policy_ver - self._target_policy_ver == self._update_target_every:
            self._target_policy.soft_update(self._policy, self._soft_update_coef)
            self._target_q_critic_net.soft_update(self._q_critic_net, self._soft_update_coef)
            self._target_policy_ver = self._policy_ver

    def get_trainer_state_dict(self) -> dict:
        return {
            "policy_state": self.get_policy_state_dict(),
            "target_policy_state": self._target_policy.get_policy_state(),
            "critic_state": self._q_critic_net.get_net_state(),
            "target_critic_state": self._target_q_critic_net.get_net_state()
        }

    def set_trainer_state_dict(self, trainer_state_dict: dict) -> None:
        self.set_policy_state_dict(trainer_state_dict["policy_state"])
        self._target_policy.set_policy_state(trainer_state_dict["target_policy_state"])
        self._q_critic_net.set_net_state(trainer_state_dict["critic_state"])
        self._target_q_critic_net.set_net_state(trainer_state_dict["target_critic_state"])
