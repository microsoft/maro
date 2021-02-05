# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import AbsActor


class Actor(AbsActor):
    def __init__(self, env, agent, state_shaper, action_shaper, experience_shaper):
        super().__init__(
            env, agent, 
            state_shaper=state_shaper, action_shaper=action_shaper, experience_shaper=experience_shaper
        )

    def roll_out(self, index, training=True):
        self.env.reset()
        metrics, event, is_done = self.env.step(None)
        while not is_done:
            agent_id = str(event.port_idx)
            state = self.state_shaper(event, self.env.snapshot_list)
            action, log_p = self.agent[agent_id].choose_action(state)
            self.experience_shaper.record(
                {"state": state, "agent_id": agent_id, "event": event, "action": action, "log_action_prob": log_p}
            )
            metrics, event, is_done = self.env.step(self.action_shaper(action, event, self.env.snapshot_list))

        exp = self.experience_shaper(self.env.snapshot_list) if training else None
        self.experience_shaper.reset()

        return exp
