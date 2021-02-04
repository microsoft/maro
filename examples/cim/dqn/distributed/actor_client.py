# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import ActorClient, TerminateRollout
from maro.utils import LogFormat, Logger


class SimpleActorClient(ActorClient):
    def __init__(
        self, env, agent_proxy, state_shaper, action_shaper, experience_shaper,
        receive_action_timeout=None, max_receive_action_attempts=None
    ):
        super().__init__(
            env, agent_proxy, 
            state_shaper=state_shaper, action_shaper=action_shaper, experience_shaper=experience_shaper,
            receive_action_timeout=receive_action_timeout, max_receive_action_attempts=max_receive_action_attempts
        )
        self._logger = Logger("actor_client", format_=LogFormat.simple, auto_timestamp=False)
    
    def roll_out(self, index, is_training=True):
        self.env.reset()
        time_step = 0
        metrics, event, is_done = self.env.step(None)
        while not is_done:
            state = self.state_shaper(event, self.env.snapshot_list)
            action = self.get_action(index, time_step, state, agent_id=str(event.port_idx))
            if isinstance(action, TerminateRollout):
                self._logger.info(f"Roll-out aborted at time step {time_step}.")
                return

            time_step += 1
            metrics, event, is_done = self.env.step(self.action_shaper(action, event, self.env.snapshot_list))
            if action is None:
                self._logger.info(
                    f"Failed to receive an action for time step {time_step}, proceed with NULL action."
                )

        exp = self.experience_shaper(self.env.snapshot_list) if is_training else None
        self.experience_shaper.reset()

        return self.env.metrics, exp
