# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np

from maro.communication import Proxy
from maro.rl import AbsRolloutExecutor, BaseActor, DecisionClient, MultiAgentWrapper
from maro.simulator import Env
from maro.utils import LogFormat, Logger

from examples.cim.dqn.components import CIMActionShaper, CIMStateShaper, CIMExperienceShaper, create_dqn_agents


class BasicRolloutExecutor(AbsRolloutExecutor):
    def __init__(self, env, agent, state_shaper, action_shaper, experience_shaper, max_null_decisions=None):
        super().__init__(
            env, agent, state_shaper=state_shaper, action_shaper=action_shaper, experience_shaper=experience_shaper
        )
        self._max_null_decisions = max_null_decisions # max number of null decisions that can be tolerated
        self._logger = Logger("actor_client", format_=LogFormat.simple, auto_timestamp=False)

    def roll_out(self, index, training=True, model_dict=None, exploration_params=None):
        self.env.reset()
        if not isinstance(self.agent, DecisionClient):
            if model_dict:
                self.agent.load_model(model_dict)
            if exploration_params:
                self.agent.set_exploration_params(exploration_params)
        time_step, null_decisions_allowed = 0, self._max_null_decisions    
        metrics, event, is_done = self.env.step(None)
        while not is_done:
            state = self.state_shaper(event, self.env.snapshot_list)
            agent_id = str(event.port_idx)
            if isinstance(self.agent, DecisionClient):
                action = self.agent.get(state, index, time_step, agent_id=agent_id)
                if action is None:
                    self._logger.info(f"Failed to receive an action for time step {time_step}, proceed with no action.")
                    if null_decisions_allowed:
                        null_decisions_allowed -= 1
                        if null_decisions_allowed == 0:
                            self._logger.info(f"Roll-out aborted due to too many null decisions.")
                            return
            else:
                action = self.agent[agent_id].choose_action(state)

            if action is not None:
                self.experience_shaper.record(
                    {"state": state, "agent_id": agent_id, "event": event, "action": action}
                )
                action = self.action_shaper(action, event, self.env.snapshot_list)
            metrics, event, is_done = self.env.step(action)
            time_step += 1

        exp = self.experience_shaper(self.env.snapshot_list) if training else None
        self.experience_shaper.reset()

        return exp


def launch(config):
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    state_shaper = CIMStateShaper(**config.env.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agent.model.output_dim)))
    experience_shaper = CIMExperienceShaper(**config.env.experience_shaping)
    
    inference_mode = config.multi_process.inference_mode
    if inference_mode == "remote":
        agent = DecisionClient(
            config.multi_process.group,
            receive_action_timeout=config.multi_process.receive_action_timeout,
            max_receive_action_attempts=config.multi_process.max_receive_action_attempts,
        )
    elif inference_mode == "local":
        config.agent.model.input_dim = state_shaper.dim
        config.agent.names = [str(agent_id) for agent_id in env.agent_idx_list]
        agent = MultiAgentWrapper(create_dqn_agents(config.agent))
    else:
        raise ValueError(f'Supported distributed training modes: "local", "remote", got {inference_mode}')
    
    executor = BasicRolloutExecutor(
        env, agent, state_shaper, action_shaper, experience_shaper,
        max_null_decisions=config.multi_process.max_null_decisions
    )
    actor = BaseActor(config.multi_process.group, executor)
    actor.run()


if __name__ == "__main__":
    from examples.cim.dqn.config import config
    launch(config)
