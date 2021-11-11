import copy
import random

import numpy as np


class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state


class OU_Noise_Exploration(object):
    """Ornstein-Uhlenbeck noise process exploration strategy"""
    def __init__(self, config):
        self.config = config
        self.noise = OU_Noise(
            self.config.action_size,
            self.config.seed,
            self.config.hyperparameters["mu"],
            self.config.hyperparameters["theta"],
            self.config.hyperparameters["sigma"]
        )

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action = action_info["action"]
        action += self.noise.sample()
        return action

    def reset(self):
        """Resets the noise process"""
        self.noise.reset()
