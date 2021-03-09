# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

import numpy as np

from maro.rl import Shaper


class TruncatedExperienceShaper(Shaper):
    def __init__(self):
        super().__init__(reward_func=None)

    def __call__(self, trajectory):
        states = trajectory["state"]
        actions = trajectory["action"]
        events = trajectory["event"]
        legal_action = trajectory["legal_action"]

        experiences_by_agent = defaultdict(lambda: defaultdict(list))
        for i in range(len(states) - 1):
            experiences = experiences_by_agent["allocator"]
            experiences["state"].append(states[i])
            experiences["action"].append(actions[i])
            experiences["reward"].append(events[i+1].profit - events[i].profit)
            experiences["next_state"].append(states[i + 1])
            experiences["next_legal_action"].append(legal_action[i + 1])

        return dict(experiences_by_agent)