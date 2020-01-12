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
                 critic_lr: float,
                 actor_lr: float,
                 theta: float,
                 sigma: float,
                 target_update_frequency: int,
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
                        sigma=sigma, 
                        theta=theta,
                        log_enable=log_enable, 
                        log_folder=log_folder)
    
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

        self_state_batch = self_state_batch.float()
        self_action_batch = self_action_batch.float()
        self_reward_batch = self_reward_batch.float()
        self_next_state_batch = self_next_state_batch.float()

        demo_state_batch = demo_state_batch.float()
        demo_action_batch = demo_action_batch.float()
        demo_reward_batch = demo_reward_batch.float()
        demo_next_state_batch = demo_next_state_batch.float()

        state_batch = torch.cat((self_state_batch, demo_state_batch))
        action_batch = torch.cat((self_action_batch, demo_action_batch))
        reward_batch = torch.cat((self_reward_batch, demo_reward_batch))
        next_state_batch = torch.cat((self_next_state_batch, demo_next_state_batch))

        super().learn(state_batch, action_batch, reward_batch, next_state_batch, current_ep)

