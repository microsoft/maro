# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.utils.exception.rl_toolkit_exception import InvalidExperience


class ExperienceSet:

    __slots__ = ["states", "actions", "rewards", "next_states"]

    def __init__(self, states: list = None, actions: list = None, rewards: list = None, next_states: list = None):
        if states is None:
            states, actions, rewards, next_states = [], [], [], []

        if not len(states) == len(actions) == len(rewards) == len(next_states):
            raise InvalidExperience("values of contents should consist of lists of the same length")
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states

    @property
    def size(self):
        return len(self.states)

    def extend(self, other):
        self.states += other.states
        self.actions += other.actions
        self.rewards += other.rewards
        self.next_states += other.next_states


class ReplayBuffer(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def to_experience_set(self):
        # print(len(self.rewards), len(self.states))
        num_complete = self.size()
        exp_set = ExperienceSet(
            states=self.states[:num_complete],
            actions=self.actions[:num_complete],
            rewards=self.rewards[:num_complete],
            next_states=self.states[1:num_complete + 1]
        )

        del self.states[:num_complete]
        del self.actions[:num_complete]
        del self.rewards[:num_complete]

        return exp_set

    def size(self):
        return min(len(self.rewards), len(self.states) - 1)
