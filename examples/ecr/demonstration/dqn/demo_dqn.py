import torch
import torch.nn as nn

from examples.ecr.q_learning.common.dqn import DQN

class DemoDQN(DQN):
    def __init__(self,
                 policy_net: nn.Module,
                 target_net: nn.Module,
                 gamma: float,
                 tau: float,
                 lr: float,
                 target_update_frequency: int,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 log_enable: bool = True, log_folder: str = './', log_dropout_p: float = 0.0,
                 dashboard_enable: bool = True, dashboard: object = None):
        super(DemoDQN, self).__init__(policy_net, target_net,
                                      gamma, tau, lr,
                                      target_update_frequency, device,
                                      log_enable, log_folder, log_dropout_p,
                                      dashboard_enable, dashboard)

    def _cal_td_error(self,
                      criterion,
                      state_batch: torch.Tensor,
                      action_batch: torch.Tensor,
                      reward_batch: torch.Tensor,
                      next_state_batch: torch.Tensor):
        state_batch = state_batch.to(self._device)
        action_batch = action_batch.to(self._device)
        reward_batch = reward_batch.to(self._device)
        next_state_batch = next_state_batch.to(self._device)
        
        policy_state_action_values = self._policy_net(state_batch).gather(1, action_batch.long())
        target_next_state_values = self._target_net(next_state_batch).max(1)[0].view(-1, 1).detach()
        expected_state_action_values = reward_batch + self._gamma * target_next_state_values

        return criterion(policy_state_action_values, expected_state_action_values)

    def learn(self,
              self_state_batch: torch.Tensor,
              self_action_batch: torch.Tensor,
              self_reward_batch: torch.Tensor,
              self_next_state_batch: torch.Tensor,
              demo_state_batch: torch.Tensor,
              demo_action_batch: torch.Tensor,
              demo_reward_batch: torch.Tensor,
              demo_next_state_batch: torch.Tensor,
              current_ep: int) -> float:
        self._policy_net.train()
        loss = 0
        # TODO: configurable criterion if needed
        criterion = torch.nn.L1Loss(reduction='mean')
        # criterion = torch.nn.MSELoss(reduction='mean')
        # TODO: weighted loss addition
        num_self = self_state_batch.size(0)
        num_demo = demo_state_batch.size(0)
        if num_self > 0:
            weight_self = num_self * 1.0 / (num_self + num_demo)
            loss += weight_self * self._cal_td_error(criterion=criterion,
                                                     state_batch=self_state_batch,
                                                     action_batch=self_action_batch,
                                                     reward_batch=self_reward_batch,
                                                     next_state_batch=self_next_state_batch)
        if num_demo > 0:
            weight_demo = num_demo * 1.0 / (num_self + num_demo)
            loss += weight_demo * self._cal_td_error(criterion=criterion,
                                                     state_batch=demo_state_batch,
                                                     action_batch=demo_action_batch,
                                                     reward_batch=demo_reward_batch,
                                                     next_state_batch=demo_next_state_batch)
        # TODO: add demo loss

        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()
        self._learning_counter += 1
        self._policy_net.eval()
        self._soft_update_target_net_parameters()