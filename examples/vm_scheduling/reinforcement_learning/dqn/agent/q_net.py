import numpy as np

import torch
import torch.nn as nn

from maro.rl import DiscreteQNet


class QNet(DiscreteQNet):
    def forward(self, states):
        states = torch.from_numpy(np.asarray(states)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return self.component(states)

    def get_action(self, states, legal_action, training=True):
        """
        Given Q-values for a batch of states and all actions, return the action index and the corresponding
        Q-values for each state.
        """
        legal_action = torch.from_numpy(np.asarray(legal_action)).to(self.device)
        illegal_action = (legal_action - 1) * 1e8
        q_for_all_actions = self.forward(states)  # (batch_size, num_actions)
        greedy_q, actions = (q_for_all_actions + illegal_action).max(dim=1)
        if training:
            return actions.detach(), greedy_q.detach()
        else:
            return actions.detach(), q_for_all_actions.detach()
