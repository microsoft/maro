# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.utils.exception.rl_toolkit_exception import InvalidExperience


class ExperienceSet:

    __slots__ = ["states", "actions", "rewards", "next_states"]

    def __init__(self, states: list, actions: list, rewards: list, next_states: list):
        if not len(states) == len(actions) == len(rewards) == len(next_states):
            raise InvalidExperience("values of contents should consist of lists of the same length")
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states

    def __len__(self):
        return len(self.states)


class Replay(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def to_experience_set(self):
        # print(len(self.rewards), len(self.states))
        num_complete = min(len(self.rewards), len(self.states) - 1)
        exp_set = ExperienceSet(
            self.states[:num_complete],
            self.actions[:num_complete],
            self.rewards[:num_complete],
            self.states[1:num_complete + 1]
        )

        del self.states[:num_complete]
        del self.actions[:num_complete]
        del self.rewards[:num_complete]

        return exp_set
