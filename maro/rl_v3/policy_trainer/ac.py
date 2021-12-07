from typing import Callable, Dict, Optional

import numpy as np
import torch

from maro.rl.utils import discount_cumsum
from maro.rl_v3.model import VNet
from maro.rl_v3.policy import DiscretePolicyGradient
from maro.rl_v3.replay_memory import FIFOReplayMemory
from maro.rl_v3.utils import TransitionBatch, ndarray_to_tensor

from .abs_trainer import SingleTrainer


class DiscreteActorCritic(SingleTrainer):
    """
    TODO: docs.
    """
    def __init__(
        self,
        name: str,
        get_v_critic_net_func: Callable[[], VNet],
        policy: DiscretePolicyGradient = None,
        replay_memory_capacity: int = 10000,
        train_batch_size: int = 128,
        grad_iters: int = 1,
        reward_discount: float = 0.9,
        lam: float = 0.9,
        clip_ratio: float = None,
        critic_loss_cls: Callable = None,
        min_logp: float = None,
        critic_loss_coef: float = 0.1,
        device: str = None,
        data_parallel: bool = False
    ) -> None:
        super(DiscreteActorCritic, self).__init__(name, device, data_parallel)

        self._replay_memory_capacity = replay_memory_capacity

        self._get_v_net_func = get_v_critic_net_func
        self._policy: Optional[DiscretePolicyGradient] = None
        self._v_critic_net: Optional[VNet] = None
        if policy is not None:
            self.register_policy(policy)

        self._train_batch_size = train_batch_size
        self._reward_discount = reward_discount
        self._lam = lam
        self._clip_ratio = clip_ratio
        self._min_logp = min_logp
        self._grad_iters = grad_iters
        self._critic_loss_coef = critic_loss_coef

        self._critic_loss_func = critic_loss_cls() if critic_loss_cls is not None else torch.nn.MSELoss()

    def _record_impl(self, policy_name: str, transition_batch: TransitionBatch) -> None:
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def _register_policy_impl(self, policy: DiscretePolicyGradient) -> None:
        assert isinstance(policy, DiscretePolicyGradient)
        self._policy = policy
        self._replay_memory = FIFOReplayMemory(
            capacity=self._replay_memory_capacity, state_dim=policy.state_dim,
            action_dim=policy.action_dim
        )
        self._v_critic_net = self._get_v_net_func()
        self._v_critic_net.to(self._device)

    def _train_step_impl(self) -> None:
        self._improve(self._get_batch())

    def get_batch_grad(
        self,
        batch: TransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Reference: https://tinyurl.com/2ezte4cr
        """
        assert scope in ("all", "actor", "critic"), \
            f"Unrecognized scope {scope}. Excepting 'all', 'actor', or 'critic'."

        states = ndarray_to_tensor(batch.states, self._device)  # s
        actions = ndarray_to_tensor(batch.actions, self._device).long()  # a

        if self._clip_ratio is not None:
            self._policy.eval()
            logps_old = self._policy.get_state_action_logps(states, actions)
        else:
            logps_old = None

        self._policy.train()
        self._v_critic_net.train()

        state_values = self._v_critic_net.v_values(states)
        values = state_values.detach().numpy()
        values = np.concatenate([values, values[-1:]])
        rewards = np.concatenate([batch.rewards, values[-1:]])

        grad_dict = {}
        if scope in ("all", "actor"):
            deltas = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]  # r + gamma * v(s') - v(s)
            advantages = ndarray_to_tensor(discount_cumsum(deltas, self._reward_discount * self._lam), self._device)

            action_probs = self._policy.get_action_probs(states)
            logps = torch.log(action_probs.gather(1, actions).squeeze())
            logps = torch.clamp(logps, min=self._min_logp, max=.0)
            if self._clip_ratio is not None:
                ratio = torch.exp(logps - logps_old)
                clipped_ratio = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
                actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
            else:
                actor_loss = -(logps * advantages).mean()  # I * delta * log pi(a|s)

            grad_dict["actor_grad"] = self._policy.get_gradients(actor_loss)

        if scope in ("all", "critic"):
            returns = ndarray_to_tensor(discount_cumsum(rewards, self._reward_discount)[:-1], self._device)
            critic_loss = self._critic_loss_func(state_values, returns)

            grad_dict["critic_grad"] = self._v_critic_net.get_gradients(critic_loss * self._critic_loss_coef)

        return grad_dict

    def _improve(self, batch: TransitionBatch) -> None:
        """
        Reference: https://tinyurl.com/2ezte4cr
        """
        for _ in range(self._grad_iters):
            grad_dict = self._get_batch_grad(batch, scope="all")
            self._policy.train()
            self._policy.apply_gradients(grad_dict["actor_grad"])
            self._v_critic_net.train()
            self._v_critic_net.apply_gradients(grad_dict["critic_grad"])

    def get_trainer_state_dict(self) -> dict:
        return {
            "critic_state": self._v_critic_net.get_net_state(),
            "policy_state": self.get_policy_state_dict()
        }

    def set_trainer_state_dict(self, trainer_state_dict: dict) -> None:
        self._v_critic_net.set_net_state(trainer_state_dict["critic_state"])
        self.set_policy_state_dict(trainer_state_dict["policy_state"])
