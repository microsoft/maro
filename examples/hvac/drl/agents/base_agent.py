from abc import abstractmethod
import os
import random
import numpy as np
import torch
import time

class BaseAgent(object):

    def __init__(self, config, env, logger):
        self.config = config
        self._set_random_seeds(config.seed)

        self.environment = env

        self.state_size = config.state_config["state_dim"]
        self.action_size = config.action_config["action_dim"]

        self.hyperparameters = config.hyperparameters

        self.total_episode_reward_so_far = 0
        self.full_episode_total_rewards = []
        self.max_episode_score_seen = -10e8

        self.rolling_return_window = 10
        self.rolling_returns = []

        self.episode_number = 0
        self.env_step_number = 0

        self.device = config.device

        self.logger = logger

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def pick_action(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    def _get_state_size(self):
        random_state = self.environment.reset()
        return random_state.size

    def _set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        # tf.set_random_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.state = self.environment.reset()
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = False
        self.total_episode_reward_so_far = 0
        if "exploration_strategy" in self.__dict__.keys():
            self.exploration_strategy.reset()

    def run_n_episodes(self):
        """Runs game to completion n times and then summarises results and saves model"""
        start = time.time()
        while self.episode_number < self.config.num_episode:
            self.reset_game()
            self.step()
            self.episode_number += 1
            self._save_result()
            self.logger.info(
                f"Episode {self.episode_number}, "
                f"Return: {self.full_episode_total_rewards[-1]: .2f}, "
                f"Max return seen: {self.max_episode_score_seen: .2f}, "
                f"Rolling return: {self.rolling_returns[-1]: .2f}, "
            )
            if self.max_episode_score_seen == self.full_episode_total_rewards[-1]:
                self.save_model()
        time_taken = time.time() - start
        return self.full_episode_total_rewards, self.rolling_returns, time_taken

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, _ = self.environment.step(action)
        self.total_episode_reward_so_far += self.reward

    def _save_result(self):
        """Saves the result of an episode of the game"""
        self.full_episode_total_rewards.append(self.total_episode_reward_so_far)
        self.rolling_returns.append(np.mean(self.full_episode_total_rewards[-self.rolling_return_window:]))
        self.max_episode_score_seen = max(self.max_episode_score_seen, self.full_episode_total_rewards[-1])

    def _take_optimization_step(self, optimizer, network, loss, clipping_norm=None):
        optimizer.zero_grad()
        loss.backward()
        self.logger.info(f"Loss -- {loss.item()}")
        if clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), clipping_norm)
        optimizer.step()

    def _soft_update_of_target_network(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def copy_model_over(from_model, to_model):
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
