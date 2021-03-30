# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

import numpy as np

from maro.rl import Shaper


class TruncatedExperienceShaper(Shaper):
    def __init__(self):
        super().__init__(reward_func=None)

    def __call__(self, trajectory, reward):
        states = trajectory["state"]
        actions = trajectory["action"]
        events = trajectory["event"]
        legal_action = trajectory["legal_action"]

        experiences_by_agent = defaultdict(lambda: defaultdict(list))
        for i in range(len(states)):
            experiences_by_agent["allocator"]["state"].append(states[i])
            experiences_by_agent["allocator"]["action"].append(actions[i])
            if legal_action[i][-1] == 1:
                experiences_by_agent["allocator"]["reward"].append(-0.1)
            else:
                experiences_by_agent["allocator"]["reward"].append(1 + reward)
            if i != len(states) - 1:
                experiences_by_agent["allocator"]["done"].append(0)
                experiences_by_agent["allocator"]["next_state"].append(states[i + 1])
                experiences_by_agent["allocator"]["next_legal_action"].append(legal_action[i + 1])
            else:
                experiences_by_agent["allocator"]["done"].append(1)
                experiences_by_agent["allocator"]["next_state"].append(np.zeros_like(states[i]))
                experiences_by_agent["allocator"]["next_legal_action"].append(np.ones_like(legal_action[i]))
        
        
        # print(experiences_by_agent["allocator"]["state"])
        # print(experiences_by_agent["allocator"]["action"])
        # print(experiences_by_agent["allocator"]["reward"])
        # print(experiences_by_agent["allocator"]["next_state"])
        # print(experiences_by_agent["allocator"]["done"])
        # print(experiences_by_agent["allocator"]["next_legal_action"])
        

        return dict(experiences_by_agent)