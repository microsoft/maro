# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple
from typing import Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from maro.rl.algorithms import AbsAlgorithm
from maro.rl.experience import ExperienceBatch
from maro.rl.model import DiscreteACNet
from maro.rl.utils import get_torch_loss_cls

ACLossInfo = namedtuple("ACLossInfo", ["actor_loss", "critic_loss", "entropy", "total_loss", "grad"])


class ActorCritic(AbsAlgorithm):
    """Actor Critic algorithm with separate policy and value models.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.
        https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

    Args:
        ac_net (DiscreteACNet): Multi-task model that computes action distributions and state values.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        critic_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for computing
            the critic loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        min_logp (float): Lower bound for clamping logP values during learning. This is to prevent logP from becoming
            very large in magnitude and cuasing stability issues. Defaults to None, which means no lower bound.
        critic_loss_coeff (float): Coefficient for critic loss in total loss. Defaults to 1.0.
        entropy_coeff (float): Coefficient for the entropy term in total loss. Defaults to None, in which case the
            total loss will not include an entropy term.
        clip_ratio (float): Clip ratio in the PPO algorithm (https://arxiv.org/pdf/1707.06347.pdf). Defaults to None,
            in which case the actor loss is calculated using the usual policy gradient theorem.
    """

    def __init__(
        self,
        ac_net: DiscreteACNet,
        reward_discount: float,
        critic_loss_cls="mse",
        min_logp: float = None,
        critic_loss_coeff: float = 1.0,
        entropy_coeff: float = None,
        clip_ratio: float = None
    ):
        if not isinstance(ac_net, DiscreteACNet):
            raise TypeError("model must be an instance of 'DiscreteACNet'")

        super().__init__()
        self.ac_net = ac_net
        self.reward_discount = reward_discount
        self.critic_loss_func = get_torch_loss_cls(critic_loss_cls)()
        self.min_logp = min_logp
        self.critic_loss_coeff = critic_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_ratio = clip_ratio

        self.device = self.ac_net.device

    def choose_action(self, states) -> Tuple[np.ndarray, np.ndarray]:
        """Return actions and log probabilities for given states."""
        self.ac_net.eval()
        with torch.no_grad():
            actions, log_p = self.ac_net.get_action(states)
        actions, log_p = actions.cpu().numpy(), log_p.cpu().numpy()
        return (actions[0], log_p[0]) if len(actions) == 1 else (actions, log_p)

    def apply(self, grad_dict: dict):
        """Apply gradients to the underlying parameterized model."""
        self.ac_net.apply(grad_dict)

    def learn(self, batch: ExperienceBatch, inplace: bool = True):
        assert self.ac_net.trainable, "ac_net needs to have at least one optimizer registered."
        self.ac_net.train()
        states, next_states = batch.data.states, batch.data.next_states
        actions = torch.from_numpy(np.asarray([act[0] for act in batch.data.actions])).to(self.device)
        log_p = torch.from_numpy(np.asarray([act[1] for act in batch.data.actions])).to(self.device)
        rewards = torch.from_numpy(np.asarray(batch.data.rewards)).to(self.device)

        action_probs, state_values = self.ac_net(states)
        state_values = state_values.squeeze()
        with torch.no_grad():
            next_state_values = self.ac_net(next_states, actor=False)[1].detach().squeeze()
        return_est = rewards + self.reward_discount * next_state_values
        advantages = return_est - state_values

        # actor loss
        log_p_new = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())  # (N,)
        log_p_new = torch.clamp(log_p_new, min=self.min_logp, max=.0)
        if self.clip_ratio is not None:
            ratio = torch.exp(log_p_new - log_p)
            clip_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            actor_loss = -(torch.min(ratio * advantages, clip_ratio * advantages)).mean()
        else:
            actor_loss = -(log_p_new * advantages).mean()

        # critic_loss
        critic_loss = self.critic_loss_func(state_values, return_est)

        # entropy
        if self.entropy_coeff is not None:
            entropy = -Categorical(action_probs).entropy().mean()
        else:
            entropy = 0

        total_loss = actor_loss + self.critic_loss_coeff * critic_loss + self.entropy_coeff * entropy

        if inplace:
            self.ac_net.step(total_loss)
            grad = None
        else:
            grad = self.ac_net.get_gradients(total_loss)

        return ACLossInfo(actor_loss, critic_loss, entropy, total_loss, grad), batch.indexes

    def set_state(self, policy_state):
        self.ac_net.load_state_dict(policy_state)

    def get_state(self):
        return self.ac_net.state_dict()
