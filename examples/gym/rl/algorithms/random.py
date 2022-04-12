import numpy as np
from gym.spaces import Space

from maro.rl.policy import RuleBasedPolicy


class RandomGymPolicy(RuleBasedPolicy):
    def __init__(self, name: str, action_space: Space) -> None:
        super(RandomGymPolicy, self).__init__(name=name)
        self._action_space = action_space

    def _rule(self, states: np.ndarray) -> object:
        n_sample = states.shape[0]
        action = [self._action_space.sample() for _ in range(n_sample)]
        return action


def get_policy(action_space: Space, name: str) -> RuleBasedPolicy:
    return RandomGymPolicy(name=name, action_space=action_space)
