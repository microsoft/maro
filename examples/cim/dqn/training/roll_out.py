# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

from maro.rl import AbsRolloutExecutor

from examples.cim.shaping_utils import get_state, get_env_action, get_reward


def get_training_data(trajectory, port_snapshots):
    states = trajectory["state"]
    actions = trajectory["action"]
    agent_ids = trajectory["agent_id"]
    events = trajectory["event"]

    exp_by_agent = defaultdict(lambda: defaultdict(list))
    for i in range(len(states) - 1):
        exp = exp_by_agent[agent_ids[i]]
        exp["state"].append(states[i])
        exp["action"].append(actions[i])
        exp["reward"].append(get_reward(events[i], port_snapshots))
        exp["next_state"].append(states[i + 1])

    return dict(exp_by_agent)


class BasicRolloutExecutor(AbsRolloutExecutor):
    def roll_out(self, index, training=True, model_dict=None, exploration_params=None):
        self.env.reset()
        trajectory = {key: [] for key in ["state", "action", "agent_id", "event"]}
        if model_dict:
            self.agent.load_model(model_dict)  
        if exploration_params:
            self.agent.set_exploration_params(exploration_params)
        metrics, event, is_done = self.env.step(None)
        while not is_done:
            state = get_state(event, self.env.snapshot_list)
            agent_id = event.port_idx
            action = self.agent[agent_id].choose_action(state)
            trajectory["state"].append(state)
            trajectory["agent_id"].append(agent_id)
            trajectory["event"].append(event)
            trajectory["action"].append(action)
            env_action = get_env_action(action, event, self.env.snapshot_list["vessels"])
            metrics, event, is_done = self.env.step(env_action)

        return get_training_data(trajectory, self.env.snapshot_list["ports"]) if training else None
