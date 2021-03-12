# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

import numpy as np

from maro.rl import AbsActor

from examples.cim.common import get_state, get_env_action, get_reward


def get_training_data(trajectory, port_snapshots):
    agent_ids = trajectory["agent_id"]
    events = trajectory["event"]
    states = trajectory["state"]
    actions = trajectory["action"]
    log_p = trajectory["log_p"]

    training_data = defaultdict(lambda: defaultdict(list))
    for i in range(len(states)):
        data = training_data[agent_ids[i]]
        data["state"].append(states[i])
        data["action"].append(actions[i])
        data["log_p"].append(log_p[i])
        data["reward"].append(get_reward(events[i], port_snapshots))
        
    for agent_id in training_data:
        for key, vals in training_data[agent_id].items():
            training_data[agent_id][key] = np.asarray(vals, dtype=np.float32 if key == "reward" else None)
    
    return training_data


class BasicActor(AbsActor):
    def roll_out(self, index, training=True):
        self.env.reset()
        trajectory = {key: [] for key in ["state", "action", "agent_id", "event", "log_p"]}
        metrics, event, is_done = self.env.step(None)
        while not is_done:
            state = get_state(event, self.env.snapshot_list)
            agent_id = event.port_idx
            action, log_p = self.agent[agent_id].choose_action(state)
            trajectory["state"].append(state)
            trajectory["agent_id"].append(agent_id)
            trajectory["event"].append(event)
            trajectory["action"].append(action)
            trajectory["log_p"].append(log_p)
            env_action = get_env_action(action, event, self.env.snapshot_list["vessels"])
            metrics, event, is_done = self.env.step(env_action)

        return get_training_data(trajectory, self.env.snapshot_list["ports"]) if training else None
