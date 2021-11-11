from abc import abstractmethod
import os
import gym
import random
import numpy as np
import torch
import time

class Base_Agent(object):

    def __init__(self, config, env, logger):
        self.config = config
        self._set_random_seeds(config.seed)
        self.environment = env
        self.environment_title = self.environment.unwrapped.id
        self.action_size = int(self.environment.action_space.shape[0])
        self.config.action_size = self.action_size

        self.state_size = int(self._get_state_size())
        self.hyperparameters = config.hyperparameters
        self.rolling_score_window = 10
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = -10e8
        self.max_episode_score_seen = -10e8
        self.episode_number = 0
        self.device = "cuda:0" if config.use_GPU else "cpu"
        self.global_step_number = 0
        self.turn_off_exploration = False
        self.model_path = os.path.join(config.checkpoint_dir, "model.pt")
        self.logger = logger

    @abstractmethod
    def step(self):
        pass

    def _get_state_size(self):
        """Gets the state_size for the gym env into the correct shape for a neural network"""
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
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(random_seed)

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        if "exploration_strategy" in self.__dict__.keys(): self.exploration_strategy.reset()

    def run_n_episodes(self):
        """Runs game to completion n times and then summarises results and saves model"""
        start = time.time()
        while self.episode_number < self.config.num_episodes:
            self.reset_game()
            self.step()
            self.save_result()
            self.logger.info(
                f"Episode {self.episode_number}, "
                f"Score: {self.game_full_episode_scores[-1]: .2f}, "
                f"Max score seen: {self.max_episode_score_seen: .2f}, "
                f"Rolling score: {self.rolling_results[-1]: .2f}, "
                f"Max rolling score seen: {self.max_rolling_score_seen: .2f}"
            )
            if len(self.game_full_episode_scores) > 0 and self.max_episode_score_seen == self.game_full_episode_scores[-1]:
                self.save_model()
        time_taken = time.time() - start
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, _ = self.environment.step(action)
        self.total_episode_score_so_far += self.reward

    def save_result(self):
        """Saves the result of an episode of the game"""
        self.game_full_episode_scores.append(self.total_episode_score_so_far)
        self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()

    def save_max_result_seen(self):
        """Updates the best episode result seen so far"""
        if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
            self.max_episode_score_seen = self.game_full_episode_scores[-1]

        if self.rolling_results[-1] > self.max_rolling_score_seen:
            if len(self.rolling_results) > self.rolling_score_window:
                self.max_rolling_score_seen = self.rolling_results[-1]

    def take_optimization_step(self, optimizer, network, loss, clipping_norm=None):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        optimizer.zero_grad()
        loss.backward()
        self.logger.info(f"Loss -- {loss.item()}")
        if clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), clipping_norm)
        optimizer.step()

    def soft_update_of_target_network(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def copy_model_over(from_model, to_model):
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
