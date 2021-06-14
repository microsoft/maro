# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from maro.rl import ActorCritic


class VMActorCritic(ActorCritic):
    def choose_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return actions and log probabilities for given states."""
        with torch.no_grad():
            self.ac_net.eval()
            action, log_p = self.ac_net.get_action(state, training)

        action, log_p = action.cpu().numpy(), log_p.cpu().numpy()
        return (action[0], log_p[0])

    def update(self):
        self.ac_net.train()
        for _ in range(self.config.train_epochs):
            experience_set = self.experience_manager.get()
            states, next_states = experience_set.states, experience_set.next_states
            if isinstance(experience_set.actions[0], tuple):
                actions = torch.from_numpy(np.asarray([act[0] for act in experience_set.actions])).to(self.device)
                log_p = torch.from_numpy(np.asarray([act[1] for act in experience_set.actions])).to(self.device)
            else:
                actions = torch.from_numpy(np.asarray(experience_set.actions)).to(self.device)
            rewards = torch.from_numpy(np.asarray(experience_set.rewards)).to(self.device)

            for _ in range(self.config.gradient_iters):
                action_probs, state_values = self.ac_net(states)
                state_values = state_values.squeeze()

                with torch.no_grad():
                    next_state_values = self.ac_net(next_states, actor=False)[1].detach().squeeze()
                # reward normalization
                rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + 1e-8)
                return_est = rewards + self.config.reward_discount * next_state_values

                advantages = return_est - state_values

                # actor loss + entropy loss
                log_p_new = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())  # (N,)
                log_p_new = torch.clamp(log_p_new, min=-20)

                if self.config.clip_ratio is not None:
                    ratio = torch.exp(log_p_new - log_p)
                    clip_ratio = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                    actor_loss = -(torch.min(ratio * advantages, clip_ratio * advantages)).mean()
                else:
                    dist = Categorical(action_probs)
                    actor_loss = -(log_p_new * advantages + 10 * dist.entropy()).mean()

                # critic_loss
                critic_loss = self.config.critic_loss_func(state_values, return_est)
                loss = critic_loss + self.config.actor_loss_coefficient * actor_loss

                self.ac_net.step(loss)

        # Empty the experience manager due to the on-policy nature of the algorithm.
        self.experience_manager.clear()
