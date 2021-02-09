# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np

from maro.communication import Proxy
from maro.rl import AbsRolloutExecutor, BaseActor, MultiAgentWrapper, RolloutClient
from maro.simulator import Env
from maro.utils import LogFormat, Logger

from examples.cim.dqn.components import CIMActionShaper, CIMStateShaper, CIMExperienceShaper, create_dqn_agents


class SimpleRolloutExecutor(AbsRolloutExecutor):
    def __init__(self, env, agent, state_shaper, action_shaper, experience_shaper):
        super().__init__(
            env, agent, 
            state_shaper=state_shaper, action_shaper=action_shaper, experience_shaper=experience_shaper
        )

    def roll_out(self, index, training=True):
        self.env.reset()
        metrics, event, is_done = self.env.step(None)
        while not is_done:
            state = self.state_shaper(event, self.env.snapshot_list)
            agent_id = str(event.port_idx)
            action = self.agent[agent_id].choose_action(state)
            self.experience_shaper.record(
                {"state": state, "agent_id": agent_id, "event": event, "action": action}
            )
            metrics, event, is_done = self.env.step(self.action_shaper(action, event, self.env.snapshot_list))

        exp = self.experience_shaper(self.env.snapshot_list) if training else None
        self.experience_shaper.reset()

        return exp


class SimpleRolloutClient(RolloutClient):
    def __init__(
        self, env, agent_proxy, state_shaper, action_shaper, experience_shaper,
        receive_action_timeout=None, max_receive_action_attempts=None, allowed_null_responses=None
    ):
        super().__init__(
            env, agent_proxy,
            state_shaper=state_shaper, action_shaper=action_shaper, experience_shaper=experience_shaper,
            receive_action_timeout=receive_action_timeout, max_receive_action_attempts=max_receive_action_attempts
        )
        # If no response occurs this many times during a roll-out episode, roll-out is aborted.
        self._allowed_null_responses = allowed_null_responses
        self._logger = Logger("actor_client", format_=LogFormat.simple, auto_timestamp=False)
    
    def roll_out(self, index, training=True):
        self.env.reset()
        allowed_null_responses = self._allowed_null_responses
        time_step = 0
        metrics, event, is_done = self.env.step(None)
        while not is_done:
            state = self.state_shaper(event, self.env.snapshot_list)
            agent_id = str(event.port_idx)
            action = self.get_action(state, index, time_step, agent_id=agent_id)

            time_step += 1
            if action is None:
                metrics, event, is_done = self.env.step(None)
                self._logger.info(f"Failed to receive an action for time step {time_step}, proceed with no action.")
                allowed_null_responses -= 1
                if allowed_null_responses == 0:
                    self._logger.info(
                        f"Received no response from learner {self._allowed_null_responses} times. Roll-out aborted."
                    )
                    return
            else:
                self.experience_shaper.record(
                    {"state": state, "agent_id": agent_id, "event": event, "action": action}
                )
                metrics, event, is_done = self.env.step(self.action_shaper(action, event, self.env.snapshot_list))
                
        exp = self.experience_shaper(self.env.snapshot_list) if training else None
        self.experience_shaper.reset()

        return exp


def launch(config):
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    state_shaper = CIMStateShaper(**config.env.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agent.model.output_dim)))
    experience_shaper = CIMExperienceShaper(**config.env.experience_shaping)
    
    inference_mode = config.multi_process.inference_mode
    redis_address = config.multi_process.redis.hostname, config.multi_process.redis.port
    if inference_mode == "remote":
        agent_proxy = Proxy(
            group_name=config.multi_process.group,
            component_type="rollout_client",
            expected_peers={"learner": 1},
            redis_address=redis_address,
            max_retries=20
        )
        executor = SimpleRolloutClient(
            env, agent_proxy, state_shaper, action_shaper, experience_shaper,
            receive_action_timeout=config.multi_process.receive_action_timeout,
            max_receive_action_attempts=config.multi_process.max_receive_action_attempts,
            allowed_null_responses=config.multi_process.allowed_null_responses
        )
    elif inference_mode == "local":
        config.agent.model.input_dim = state_shaper.dim
        config.agent.names = [str(agent_id) for agent_id in env.agent_idx_list]
        agent = MultiAgentWrapper(create_dqn_agents(config.agent))
        executor = SimpleRolloutExecutor(env, agent, state_shaper, action_shaper, experience_shaper)
    else:
        raise ValueError(f'Supported distributed training modes: "local", "remote", got {inference_mode}')

    proxy = Proxy(
        group_name=config.multi_process.group,
        component_type="actor",
        expected_peers={"learner": 1},
        redis_address=redis_address,
        max_retries=20
    )
    
    actor = BaseActor(executor, proxy)
    actor.run()


if __name__ == "__main__":
    from examples.cim.dqn.config import config
    launch(config)
