import datetime
import os

from maro.rl import Scheduler
from maro.simulator import Env
from maro.utils import Logger

from examples.cim.gnn.general import simulation_logger, training_logger
from examples.cim.gnn.components import (
    DiscreteActionShaper, GNNAgentManager, GNNExperienceShaper, GNNLearner, GNNStateShaper,
    create_gnn_agent, decision_cnt_analysis, load_config, return_scaler, save_code, save_config
)


def launch(config):
    # Create a demo environment to retrieve environment information.
    simulation_logger.info("Getting experience quantity estimates for each (port, vessel) pair...")
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    exp_per_ep = decision_cnt_analysis(env, pv=True, buffer_size=8)
    config.agent.exp_per_ep = {
        agent_id: cnt * config.main_loop.train_freq for agent_id, cnt in exp_per_ep.items()
    }
    simulation_logger.info(config.agent.exp_per_ep)

    # Add some buffer to prevent overlapping.
    config.env.return_scaler, _ = return_scaler(
        env, tick=config.env.durations, gamma=config.agent.algorithm.reward_discount
    )
    simulation_logger.info(f"Return value will be scaled down by a factor of {config.env.return_scaler}")

    save_config(config, os.path.join(config.log.path, "config.yml"))
    # save_code("examples/cim/gnn", config.log.path)

    port_mapping = env.summary["node_mapping"]["ports"]
    vessel_mapping = env.summary["node_mapping"]["vessels"]

    # Create a mock gnn state shaper.
    static_code_list, dynamic_code_list = list(port_mapping.values()), list(vessel_mapping.values())
    state_shaper = GNNStateShaper(
        static_code_list, dynamic_code_list, config.env.durations, config.agent.model.feature,
        sequence_buffer_size=config.agent.model.sequence_buffer_size, max_value=env.configs["total_containers"]
    )
    state_shaper.compute_static_graph_structure(env)
    action_shaper = DiscreteActionShaper(config.agent.model.action_dim)
    experience_shaper = GNNExperienceShaper(
        static_code_list, dynamic_code_list, config.env.durations, state_shaper,
        scale_factor=config.env.return_scaler, time_slot=config.agent.algorithm.td_steps,
        discount_factor=config.agent.algorithm.reward_discount
    )

    # Create an agent_manager.
    config.agent.num_static_nodes = len(static_code_list)
    config.agent.num_dynamic_nodes = len(dynamic_code_list)
    config.agent.algorithm.p2p_adj = state_shaper.p2p_static_graph
    config.agent.model.port_feature_dim = state_shaper.get_input_dim("p")
    config.agent.model.vessel_feature_dim = state_shaper.get_input_dim("v")
    agent = create_gnn_agent(config.agent)
    agent_manager = GNNAgentManager(
        agent, state_shaper=state_shaper, action_shaper=action_shaper, experience_shaper=experience_shaper
    )
    # Learner function for training and testing.
    learner = GNNLearner(
        env, agent_manager, Scheduler(config.main_loop.max_episode), 
        train_freq=config.main_loop.train_freq,
        model_save_freq=config.main_loop.model_save_freq,
        logger=simulation_logger
    )
    learner.learn()


if __name__ == "__main__":
    from examples.cim.gnn.general import config
    launch(config)
