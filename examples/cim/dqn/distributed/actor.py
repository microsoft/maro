# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import ActorClient
from maro.utils import LogFormat, Logger


class SimpleActorClient(ActorClient):
    def __init__(
        self, env, agent_proxy, state_shaper, action_shaper, experience_shaper
        receive_action_timeout=None, max_receive_action_attempts=None
    ):
        super().__init__(
            env, agent_proxy, 
            state_shaper=state_shaper, action_shaper=action_shaper, experience_shaper=experience_shaper,
            receive_action_timeout=receive_action_timeout, max_receive_action_attempts=max_receive_action_attempts
        )
        self._logger = Logger("actor_client", format_=LogFormat.simple, auto_timestamp=False)
    
    def roll_out(self, is_training: bool = True):
        self.logger.info(f"Rolling out for ep-{ep}...")
        self.env.reset()
        metrics, event, is_done = self.env.step(None)
        while not is_done:
            action = self.get_action(*self.state_shaper(event, self.env.snapshot_list))
            if isinstance(action, TerminateEpisode):
                self.logger.info(f"Roll-out aborted at time step {self.agent.time_step}.")
                return

            metrics, event, is_done = self.env.step(self.action_shaper(action, event, self.env.snapshot_list))
            if not action:
                self.logger.info(
                    f"Failed to receive an action for time step {self.agent.time_step}, "
                    f"proceed with NULL action."
                )

        self.logger.info(f"Roll-out finished for ep-{self._ep}")
