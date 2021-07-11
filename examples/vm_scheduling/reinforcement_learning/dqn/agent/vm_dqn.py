# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import torch
import numpy as np

from maro.rl import DQN


class VMDQN(DQN):
    def choose_action(self, state: np.ndarray, training: bool = True) -> Union[int, np.ndarray]:
        """Return action."""
        with torch.no_grad():
            self.q_net.eval()
            action, q_value = self.q_net.get_action(state, training)

        action = action.cpu().numpy()
        q_value = q_value.cpu().numpy()
        if training:
            return action[0]
        else:
            return action[0], q_value[0]

    def update(self):
        assert self.q_net.trainable, "q_net needs to have at least one optimizer registered."
        self.q_net.train()
        for _ in range(self.config.train_epochs):
            # sample from the replay memory
            experience_set = self.experience_manager.get()
            states, next_states = experience_set.states, experience_set.next_states
            info = np.asarray(experience_set.info)
            next_legal_action, gamma = info[:, :-1], info[:, -1]
            gamma = torch.from_numpy(gamma).to(self.device)
            actions = torch.from_numpy(np.asarray(experience_set.actions)).to(self.device)
            rewards = torch.from_numpy(np.asarray(experience_set.rewards)).to(self.device)
            if self.config.double:
                for _ in range(self.config.gradient_iters):
                    # get target Q values
                    with torch.no_grad():
                        actions_by_eval_q_net = self.q_net.get_action(next_states, next_legal_action)[0]
                        next_q_values = self.target_q_net.q_values(next_states, actions_by_eval_q_net)

                    # gradient steps
                    q_values = self.q_net.q_values(states, actions)
                    loss = self._get_td_errors(
                        q_values, next_q_values, rewards, gamma, loss_func=self.config.loss_func
                    )
                    self.q_net.step(loss.mean())
            else:
                # get target Q values
                with torch.no_grad():
                    next_q_values = self.target_q_net.get_action(next_states, next_legal_action)[1]  # (N,)

                # gradient steps
                for _ in range(self.config.gradient_iters):
                    q_values = self.q_net.q_values(states, actions)
                    loss = self._get_td_errors(
                        q_values, next_q_values, rewards, gamma, loss_func=self.config.loss_func
                    )
                    self.q_net.step(loss.mean())

            self._training_counter += 1
            if self._training_counter % self.config.target_update_freq == 0:
                self.target_q_net.soft_update(self.q_net, self.config.soft_update_coefficient)

    @staticmethod
    def _get_td_errors(
        q_values, next_q_values, rewards, gamma, loss_func
    ):
        target_q_values = (rewards + gamma * next_q_values).detach()  # (N,)
        return loss_func(q_values, target_q_values)
