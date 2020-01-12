from collections import defaultdict
from tqdm import tqdm

import numpy as np
import random
import torch

from examples.ecr.q_learning.common.agent import Agent
from maro.utils import AbsExperiencePool


class DemoDQNAgent(Agent):
    def __init__(self,
                 agent_name: str,
                 topology: str,
                 port_idx2name: dict,
                 vessel_idx2name: dict,
                 algorithm,
                 state_shaping,
                 action_shaping,
                 reward_shaping: str,
                 experience_pool: AbsExperiencePool,
                 demo_experience_pool: AbsExperiencePool,
                 training_config,
                 agent_idx_list,
                 log_enable: bool,
                 log_folder: str,
                 dashboard_enable: bool,
                 dashboard = None
                 ):
        super(DemoDQNAgent, self).__init__(agent_name=agent_name,
                                           topology=topology,
                                           port_idx2name=port_idx2name,
                                           vessel_idx2name=vessel_idx2name,
                                           algorithm=algorithm,
                                           experience_pool=experience_pool,
                                           state_shaping=state_shaping,
                                           action_shaping=action_shaping,
                                           reward_shaping=reward_shaping,
                                           batch_num=-1,
                                           batch_size=training_config.batch_size,
                                           min_train_experience_num=training_config.minimum_experience_num,
                                           agent_idx_list=agent_idx_list,
                                           log_enable=log_enable,
                                           log_folder=log_folder,
                                           dashboard_enable=dashboard_enable,
                                           dashboard=dashboard
                                           )
        self._training_config = training_config
        self._demo_experience_pool = demo_experience_pool
        self._experience_ratio_dict = self._parse_demo_experience_ratio(training_config.demo_experience_ratio)

    def _parse_demo_experience_ratio(self, ep_ratio_list: list):
        self_demo_ratio = {}
        self_demo_ratio = defaultdict(lambda: (1.0, 0.0), self_demo_ratio)

        ep_idx = 0
        for ep_ratio_pair in ep_ratio_list:
            separator_index = ep_ratio_pair.index(':')
            episode = int(ep_ratio_pair[:separator_index])
            ratio = float(ep_ratio_pair[separator_index + 1:])
            while(ep_idx < episode):
                self_demo_ratio[ep_idx] = (1 - ratio, ratio)
                ep_idx += 1

        return self_demo_ratio

    def _cal_batch_num(self, config, self_ratio, demo_ratio):
        if config.enable_constant:
            return config.constant
        times = config.experience_times
        self_batch_num = round(self._experience_pool.size['state'] * times / max(self._batch_size * self_ratio, 1))
        demo_batch_num = round(self._demo_experience_pool.size['state'] * times / max(self._batch_size * demo_ratio, 1))
        return min(self_batch_num, demo_batch_num)

    def meet_training_condition(self, current_ep):
        self_ratio, demo_ratio = self._experience_ratio_dict[current_ep]
        return self._experience_pool.size['state'] >= self._min_train_experience_num * self_ratio \
            and self._demo_experience_pool.size['state'] >= self._min_train_experience_num * demo_ratio

    def train(self, current_ep: int):
        if not self.meet_training_condition(current_ep):
            return 0

        self_ratio, demo_ratio = self._experience_ratio_dict[current_ep]
        self_experience_num = int(self._batch_size * self_ratio)
        demo_experience_num = int(self._batch_size * demo_ratio)

        batch_num = self._cal_batch_num(self._training_config.batch_num_per_training, self_ratio, demo_ratio)

        pbar = tqdm(range(batch_num))
        for i in pbar:
            pbar.set_description(f'Agent {self._agent_name} batch training {i + 1}/{batch_num}')

            # Get Self Experience Data
            self_idxs = range(self._experience_pool.size['state'])
            self_sampled_idxs = random.choices(self_idxs, k=self_experience_num)
            self_sampled_dict = self._experience_pool.get( \
                category_idx_batches=[('state', self_sampled_idxs), ('action', self_sampled_idxs), ('reward', self_sampled_idxs), ('next_state', self_sampled_idxs)])
            self_state_batch = torch.from_numpy(np.array(self_sampled_dict['state'])).view(-1, self._algorithm.policy_net.input_dim)
            self_action_batch = torch.from_numpy(np.array(self_sampled_dict['action'])).view(-1, 1)
            self_reward_batch = torch.from_numpy(np.array(self_sampled_dict['reward'])).view(-1, 1)
            self_next_state_batch = torch.from_numpy(np.array(self_sampled_dict['next_state'])).view(-1, self._algorithm.policy_net.input_dim)

            # Get Demo Experience Data
            demo_idxs = range(self._demo_experience_pool.size['state'])
            demo_sampled_idxs = random.choices(demo_idxs, k=demo_experience_num)
            demo_sampled_dict = self._demo_experience_pool.get( \
                category_idx_batches=[('state', demo_sampled_idxs), ('action', demo_sampled_idxs), ('reward', demo_sampled_idxs), ('next_state', demo_sampled_idxs)])
            demo_state_batch = torch.from_numpy(np.array(demo_sampled_dict['state'])).view(-1, self._algorithm.policy_net.input_dim)
            demo_action_batch = torch.from_numpy(np.array(demo_sampled_dict['action'])).view(-1, 1)
            demo_reward_batch = torch.from_numpy(np.array(demo_sampled_dict['reward'])).view(-1, 1)
            demo_next_state_batch = torch.from_numpy(np.array(demo_sampled_dict['next_state'])).view(-1, self._algorithm.policy_net.input_dim)

            # Learn and Update Algorithm Model
            self._algorithm.learn(self_state_batch, self_action_batch, self_reward_batch, self_next_state_batch, \
                                  demo_state_batch, demo_action_batch, demo_reward_batch, demo_next_state_batch, \
                                  current_ep=current_ep)

    def fulfill_cache(self, agent_idx_list: [int], snapshot_list, current_ep: int):
        self.calculate_offline_rewards(snapshot_list=snapshot_list, current_ep=current_ep)

    def put_experience(self):
        self.store_experience()

    def clear_cache(self):
        self._clear_cache()