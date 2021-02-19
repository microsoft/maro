# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl import AbsRolloutExecutor
from maro.simulator import Env
from maro.utils import DummyLogger

from examples.cim.shaping_utils import get_env_action


class BasicRolloutExecutor(AbsRolloutExecutor):
    def __init__(self, env, agent, state_shaper, experience_shaper, max_null_actions=None, logger=None):
        super().__init__(env, agent)
        self.state_shaper = state_shaper
        self.experience_shaper = experience_shaper
        self._max_null_actions = max_null_actions
        self._logger = logger if logger else DummyLogger()

    def roll_out(self, index, training=True):
        self.env.reset()
        time_step, null_decisions_allowed, logs = 0, 0, []
        metrics, event, is_done = self.env.step(None)
        while not is_done:
            state = self.state_shaper(event, self.env.snapshot_list)
            action_info = self.agent.get(state, index, time_step)
            if action_info is None:
                self._logger.info(f"Failed to receive an action for time step {time_step}, proceed with no action.")
                if null_decisions_allowed:
                    null_decisions_allowed -= 1
                    if null_decisions_allowed == 0:
                        self._logger.info(f"Roll-out aborted due to too many null decisions.")
                        return
                env_action = None
            else:
                self.experience_shaper.record(event, action_info[0], state)
                env_action = get_env_action(
                    action_info[0], event, self.env.snapshot_list["vessels"],
                    finite_vessel_space=False, has_early_discharge=False
                )
            logs.append([
                event.tick, event.port_idx, event.vessel_idx, action_info, env_action,
                event.action_scope.load, event.action_scope.discharge
            ])
            time_step += 1
            metrics, event, is_done = self.env.step(env_action)

        self.state_shaper.end_ep_callback(self.env.snapshot_list)
        exp = self.experience_shaper(self.env.snapshot_list) if training else None
        self.experience_shaper.reset()

        return exp, logs
