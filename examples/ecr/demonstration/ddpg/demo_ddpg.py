from examples.ecr.ddpg.ddpg import DDPG

from torch import nn

import torch

class DemoDDPG(DDPG):
    def __init__(self,
                 actor_policy_net: nn.Module, 
                 actor_target_net: nn.Module,
                 critic_policy_net: nn.Module, 
                 critic_target_net: nn.Module,
                 gamma: float,
                 tau: float,
                 target_update_frequency: int,
                 critic_lr: float,
                 actor_lr: float,
                 theta: float,
                 sigma: float,
                 demo_action_reward_augment_ratio: float,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 log_enable: bool = True, 
                 log_folder: str = './'):
    
        super().__init__(actor_policy_net=actor_policy_net, 
                        actor_target_net=actor_target_net,
                        critic_policy_net=critic_policy_net, 
                        critic_target_net=critic_target_net,
                        gamma=gamma, 
                        tau=tau, 
                        target_update_frequency=target_update_frequency, 
                        critic_lr=critic_lr,
                        actor_lr=actor_lr, 
                        theta=theta,
                        sigma=sigma, 
                        log_enable=log_enable, 
                        log_folder=log_folder)
        self._demo_action_reward_augment_ratio = demo_action_reward_augment_ratio
    
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

        self_state_batch = self_state_batch.float().to(self._device)
        self_action_batch = self_action_batch.float().to(self._device)
        self_reward_batch = self_reward_batch.float().to(self._device)
        self_next_state_batch = self_next_state_batch.float().to(self._device)

        demo_state_batch = demo_state_batch.float().to(self._device)
        demo_action_batch = demo_action_batch.float().to(self._device)
        demo_reward_batch = demo_reward_batch.float().to(self._device)
        demo_next_state_batch = demo_next_state_batch.float().to(self._device)
        
        """ Critic """
        if self_state_batch.shape[0] > 0:
            self_current_Q_values = self._critic_policy_net(self_state_batch, self_action_batch)
            self_target_actions = self._actor_target_net(self_next_state_batch)#.clamp(-self.action_lim, self.action_lim)
            self_next_Q_values = self._critic_target_net(self_next_state_batch, self_target_actions).detach()
            self_target_Q_values = self_reward_batch + (self._gamma * self_next_Q_values)
        
        # print(self_target_Q_values)

        if demo_state_batch.shape[0] > 0:
            demo_current_Q_values = self._critic_policy_net(demo_state_batch, demo_action_batch)
            demo_target_actions = self._actor_target_net(demo_next_state_batch)#.clamp(-self.action_lim, self.action_lim)
            demo_next_Q_values = self._critic_target_net(demo_next_state_batch, demo_target_actions).detach()
            demo_target_Q_values = demo_reward_batch * self._demo_action_reward_augment_ratio + (self._gamma * demo_next_Q_values)

        # print(demo_target_Q_values)
        critic_loss = F.mse_loss(torch.cat((self_current_Q_values, demo_current_Q_values)), \
            torch.cat((self_target_Q_values, demo_target_Q_values))).mean()

        for action_index, q_values in zip(self_action_batch, self_current_Q_values):
            self._Q_value_logger.debug(f'{current_ep}, {action_index.item()}, {q_values.item()}')

        for action_index, q_values in zip(demo_action_batch, demo_current_Q_values):
            self._Q_value_logger.debug(f'{current_ep}, {action_index.item()}, {q_values.item()}')
        
        # Optimize the critic
        self._critic_policy_net.train()
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        # for param in self._Critic_policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self._critic_optimizer.step()
        self._critic_policy_net.eval()

        """ Actor """
        actor_loss = -self._critic_policy_net(torch.cat((self_state_batch, demo_state_batch)), self._actor_policy_net(torch.cat((self_state_batch, demo_state_batch)))).mean()

        # Optimize the actor
        self._actor_policy_net.train()
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        # for param in self._Actor_policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self._actor_optimizer.step()
        self._actor_policy_net.eval()

        '''update target critic and actor'''

        self._learning_counter += 1
        self._actor_policy_net.eval()
        self._critic_policy_net.eval()
        self._update_target_model()

        if self._log_enable:
            # self._critic_policy_net.log_model_parameters(current_ep, self._learning_counter)
            # self._actor_policy_net.log_model_parameters(current_ep, self._learning_counter)
            self._actor_loss_logger.debug(f'{current_ep},{self._learning_counter},actor,{actor_loss.item()}')
            self._critic_loss_logger.debug(f'{current_ep},{self._learning_counter},critic,{critic_loss.item()}')

        return critic_loss.item()
