# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple

import numpy as np
import torch

from maro.rl.experience import ExperienceManager
from maro.rl.model import DiscreteACNet
from maro.rl.policy import AbsCorePolicy
from maro.rl.utils import get_torch_loss_cls


class ActorCriticConfig:
    """Configuration for the Actor-Critic algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        train_epochs (int): Number of training epochs per call to ``update()``. Defaults to 1.
        gradient_iters (int): Number of gradient steps for each mini-batch. Defaults to 1.
        critic_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for computing
            the critic loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        actor_loss_coefficient (float): The coefficient for actor loss in the total loss function, e.g.,
            loss = critic_loss + ``actor_loss_coefficient`` * actor_loss. Defaults to 1.0.
        clip_ratio (float): Clip ratio in the PPO algorithm (https://arxiv.org/pdf/1707.06347.pdf). Defaults to None,
            in which case the actor loss is calculated using the usual policy gradient theorem.
    """
    __slots__ = [
        "reward_discount", "train_epochs", "gradient_iters", "critic_loss_func", "actor_loss_coefficient", "clip_ratio"
    ]

    def __init__(
        self,
        reward_discount: float,
        train_epochs: int = 1,
        gradient_iters: int = 1,
        critic_loss_cls="mse",
        actor_loss_coefficient: float = 1.0,
        clip_ratio: float = None
    ):
        self.reward_discount = reward_discount
        self.train_epochs = train_epochs
        self.gradient_iters = gradient_iters
        self.critic_loss_func = get_torch_loss_cls(critic_loss_cls)()
        self.actor_loss_coefficient = actor_loss_coefficient
        self.clip_ratio = clip_ratio


class ActorCritic(AbsCorePolicy):
    """Actor Critic algorithm with separate policy and value models.

    In accordance with its on-policy nature, the experiences manager is emptied at the end of each ``update()`` call.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.
        https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

    Args:
        name (str): Policy name.
        ac_net (DiscreteACNet): Multi-task model that computes action distributions and state values.
        experience_manager (ExperienceManager): An experience manager for storing and retrieving experiences
            for training.
        config: Configuration for the AC algorithm.
        update_trigger (int): Minimum number of new experiences required to trigger an ``update`` call. Defaults to 1.
        warmup (int): Minimum number of experiences in the experience memory required to trigger an ``update`` call.
            Defaults to 1.
    """

    def __init__(
        self,
        name: str,
        ac_net: DiscreteACNet,
        experience_manager: ExperienceManager,
        config: ActorCriticConfig,
        update_trigger: int = 1,
        warmup: int = 1
    ):
        if not isinstance(ac_net, DiscreteACNet):
            raise TypeError("model must be an instance of 'DiscreteACNet'")

        super().__init__(name, experience_manager, update_trigger=update_trigger, warmup=warmup)
        self.ac_net = ac_net
        self.config = config
        self.device = self.ac_net.device

    def choose_action(self, states) -> Tuple[np.ndarray, np.ndarray]:
        """Return actions and log probabilities for given states."""
        with torch.no_grad():
            actions, log_p = self.ac_net.get_action(states)
        actions, log_p = actions.cpu().numpy(), log_p.cpu().numpy()
        return (actions[0], log_p[0]) if len(actions) == 1 else (actions, log_p)

    def update(self):
        self.ac_net.train()
        for _ in range(self.config.train_epochs):
            experience_set = self.experience_manager.get()
            states, next_states = experience_set.states, experience_set.next_states
            actions = torch.from_numpy(np.asarray([act[0] for act in experience_set.actions])).to(self.device)
            log_p = torch.from_numpy(np.asarray([act[1] for act in experience_set.actions])).to(self.device)
            rewards = torch.from_numpy(np.asarray(experience_set.rewards)).to(self.device)

            for _ in range(self.config.gradient_iters):
                action_probs, state_values = self.ac_net(states)
                state_values = state_values.squeeze()
                with torch.no_grad():
                    next_state_values = self.ac_net(next_states, actor=False)[1].detach().squeeze()
                return_est = rewards + self.config.reward_discount * next_state_values
                advantages = return_est - state_values

                # actor loss
                log_p_new = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())  # (N,)
                if self.config.clip_ratio is not None:
                    ratio = torch.exp(log_p_new - log_p)
                    clip_ratio = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                    actor_loss = -(torch.min(ratio * advantages, clip_ratio * advantages)).mean()
                else:
                    actor_loss = -(log_p_new * advantages).mean()

                # critic_loss
                critic_loss = self.config.critic_loss_func(state_values, return_est)
                loss = critic_loss + self.config.actor_loss_coefficient * actor_loss

                self.ac_net.step(loss)

        # Empty the experience manager due to the on-policy nature of the algorithm.
        self.experience_manager.clear()

    def set_state(self, policy_state):
        self.ac_net.load_state_dict(policy_state)

    def get_state(self):
        return self.ac_net.state_dict()
