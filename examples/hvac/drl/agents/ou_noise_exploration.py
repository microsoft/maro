from utilities.OU_Noise import OU_Noise

class OU_Noise_Exploration(object):
    """Ornstein-Uhlenbeck noise process exploration strategy"""
    def __init__(self, config):
        self.config = config
        self.noise = OU_Noise(self.config.action_size, self.config.seed, self.config.hyperparameters["mu"],
                              self.config.hyperparameters["theta"], self.config.hyperparameters["sigma"])

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action = action_info["action"]
        action += self.noise.sample()
        return action

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        self.noise.reset()
