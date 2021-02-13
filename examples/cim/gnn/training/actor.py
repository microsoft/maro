# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.communication import Proxy
from maro.rl import AbsRolloutExecutor, BaseActor, DecisionClient
from maro.simulator import Env
from maro.utils import LogFormat, Logger

from examples.cim.gnn.components import GNNStateShaper, DiscreteActionShaper, GNNExperienceShaper
from examples.cim.gnn.general import simulation_logger
from examples.cim.gnn.training.utils import fix_seed, return_scaler


class BasicRolloutExecutor(AbsRolloutExecutor):
    def __init__(self, env, agent, state_shaper, action_shaper, experience_shaper):
        super().__init__(
            env, agent, state_shaper=state_shaper, action_shaper=action_shaper, experience_shaper=experience_shaper
        )

    def roll_out(self, index, training=True):
        self.env.reset()
        time_step, logs = 0, []
        metrics, event, is_done = self.env.step(None)
        while not is_done:
            state = self.state_shaper(event, self.env.snapshot_list)
            if isinstance(self.agent, DecisionClient):
                action, _ = self.agent.get(state, index, time_step)
            else:
                action, _ = self.agent.choose_action(state)
            self.experience_shaper.record(event, action, state)
            env_action = self.action_shaper(action, event)
            logs.append([
                event.tick, event.port_idx, event.vessel_idx, action, env_action,
                event.action_scope.load, event.action_scope.discharge
            ])
            time_step += 1
            metrics, event, is_done = self.env.step(env_action)

        self.state_shaper.end_ep_callback(self.env.snapshot_list)
        exp = self.experience_shaper(self.env.snapshot_list) if training else None
        self.experience_shaper.reset()

        return exp, logs


def launch(config):
    # Create a demo environment to retrieve environment information.
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)

    # Add some buffer to prevent overlapping.
    scale_factor, _ = return_scaler(env, config.env.durations, config.agent.hyper_params.reward_discount)
    simulation_logger.info(f"Return values will be scaled down by a factor of {scale_factor}")

    static_code_list = list(env.summary["node_mapping"]["ports"].values())
    dynamic_code_list = list(env.summary["node_mapping"]["vessels"].values())

    # Create shapers
    state_shaper = GNNStateShaper(
        static_code_list, dynamic_code_list, config.env.durations, config.agent.model.feature,
        sequence_buffer_size=config.agent.model.sequence_buffer_size, max_value=env.configs["total_containers"]
    )
    state_shaper.compute_static_graph_structure(env)
    action_shaper = DiscreteActionShaper(config.agent.model.action_dim)
    experience_shaper = GNNExperienceShaper(
        static_code_list, dynamic_code_list, config.env.durations, state_shaper,
        scale_factor=scale_factor, time_slot=config.agent.hyper_params.td_steps,
        discount_factor=config.agent.hyper_params.reward_discount
    )

    inference_mode = config.multi_process.inference_mode
    if inference_mode == "remote":
        agent = DecisionClient(config.multi_process.group)
    elif inference_mode == "local":
        config.agent.num_static_nodes = len(static_code_list)
        config.agent.num_dynamic_nodes = len(dynamic_code_list)
        config.agent.hyper_params.p2p_adj = state_shaper.p2p_static_graph
        config.agent.model.port_feature_dim = state_shaper.get_input_dim("p")
        config.agent.model.vessel_feature_dim = state_shaper.get_input_dim("v")
        agent = create_gnn_agent(config.agent)
    else:
        raise ValueError(f'Supported distributed training modes: "local", "remote", got {inference_mode}')
    
    executor = BasicRolloutExecutor(env, agent, state_shaper, action_shaper, experience_shaper)
    actor = BaseActor(config.multi_process.group, executor)
    actor.run()


if __name__ == "__main__":
    from examples.cim.gnn.general import config
    launch(config)
