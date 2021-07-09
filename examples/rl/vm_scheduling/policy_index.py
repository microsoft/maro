# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np

from maro.rl.exploration import EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler


cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
from ac import get_ac_policy
from env_wrapper import AGENT_IDS, env_config

update_trigger = {name: 128 for name in AGENT_IDS}
warmup = {name: 1 for name in AGENT_IDS}

# use agent IDs as policy names since each agent uses a separate policy
policy_func_index = {name: get_ac_policy for name in AGENT_IDS}
agent2policy = {name: name for name in AGENT_IDS}


class VMExploration(EpsilonGreedyExploration):
    def __call__(self, action_index, legal_action):
        if isinstance(action_index, np.ndarray):
            return np.array([self._get_exploration_action(act) for act in action_index])
        else:
            return self._get_exploration_action(action_index, legal_action)

    def _get_exploration_action(self, action_index, legal_action):
        assert (action_index < self._num_actions), f"Invalid action: {action_index}"
        return action_index if np.random.random() > self.epsilon else np.random.choice(np.where(legal_action == 1)[0])


"""
def __call__(self, action_index: Union[int, np.ndarray]):
    if isinstance(action_index, np.ndarray):
        return np.array([self._get_exploration_action(act) for act in action_index])
    else:
        return self._get_exploration_action(action_index)

def _get_exploration_action(self, action_index):
    assert (action_index < self._num_actions), f"Invalid action: {action_index}"
    return action_index if np.random.random() > self.epsilon else np.random.choice(self._num_actions)
"""

exploration_config = {
    "last_ep": 400,
    "initial_value": 0.4,
    "final_value": 0.0,
    "splits": [[100, 0.32]]
}

def get_exploration():
    epsilon_greedy = EpsilonGreedyExploration(num_actions=env_config["wrapper"]["num_actions"])
    epsilon_greedy.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        **exploration_config
    )
    return epsilon_greedy


exploration_func_index = {f"EpsilonGreedy": get_exploration}
agent2exploration = {name: "EpsilonGreedy" for name in AGENT_IDS}
