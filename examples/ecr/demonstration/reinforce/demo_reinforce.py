from torch.distributions import Categorical

import torch
import torch.nn as nn

from examples.ecr.reinforce.reinforce import Reinforce

class DemoReinforce(Reinforce):
    def __init__(self,
                 policy_net: nn.Module,
                 lr: float,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 log_enable: bool = True, log_folder: str = './',
                 dashboard_enable: bool = True, dashboard: object = None):
        super(DemoReinforce, self).__init__(policy_net, lr, device,
                                      log_enable, log_folder,
                                      dashboard_enable, dashboard)

    def learn(self,
              self_state_batch: torch.Tensor,
              self_action_batch: torch.Tensor,
              self_reward_batch: torch.Tensor,
              demo_state_batch: torch.Tensor,
              demo_action_batch: torch.Tensor,
              demo_reward_batch: torch.Tensor,
              current_ep: int) -> float:

        self._policy_net.train()
        loss = 0
        if self_state_batch.size(0) > 0:
            for self_state, self_action, self_reward in zip(self_state_batch, self_action_batch, self_reward_batch):
                self_probs = self._policy_net.forward(self_state)
                self_m = Categorical(self_probs)
                loss += -self_m.log_prob(self_action) * self_reward
        
        if demo_state_batch.size(0) > 0:
            for demo_state, demo_action, demo_reward in zip(demo_state_batch, demo_action_batch, demo_reward_batch):
                demo_probs = self._policy_net.forward(demo_state)
                demo_m = Categorical(demo_probs)
                loss += -demo_m.log_prob(demo_action) * demo_reward
        loss = loss.sum()
        loss /= self_state_batch.size(0) + demo_state_batch.size(0)
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()
        self._learning_counter += 1
        self._policy_net.eval()