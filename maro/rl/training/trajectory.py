# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict


class Trajectory(object):
    def __init__(self, env):
        self.env = env
        self.trajectory = defaultdict(list)

    def get_state(self, event) -> dict:
        pass

    def get_action(self, action_by_agent, event) -> dict:
        pass

    def get_reward(self) -> float:
        pass

    def on_env_feedback(self):
        pass

    def on_finish(self):
        pass

    def reset(self):
        self.trajectory = defaultdict(list)
