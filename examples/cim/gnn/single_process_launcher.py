import datetime
import os

from maro.rl import Scheduler
from maro.simulator import Env
from maro.utils import Logger

from examples.cim.general import simulation_logger, training_logger
from examples.cim.gnn.components import (
    GNNLearner, GNNStateShaper, ParallelActor, GNNAgentManger,
    create_gnn_agent, decision_cnt_analysis, load_config, return_scaler, save_code, save_config
)


def launch(config):
    # Create a demo environment to retrieve environment information.
    simulation_logger.info("Approximating the experience quantity for each agent...")
    env = Env(**config.env.param)
    exp_per_ep = decision_cnt_analysis(env, pv=True, buffer_size=8)
    config.agent.exp_per_ep = {
        agent_id: cnt * config.main_loop.train_freq for agent_id, cnt in exp_per_ep.items()
    }
    simulation_logger.info(config.agent.exp_per_ep)

    # Add some buffer to prevent overlapping.
    config.env.return_scaler, _ = return_scaler(
        env, tick=config.env.param.durations, gamma=config.agent.algorithm.reward_discount
    )
    simulation_logger.info(f"Return value will be scaled down by a factor of {config.env.return_scaler}")

    save_config(config, os.path.join(config.log.path, "config.yml"))
    save_code("examples/cim/gnn", config.log.path)

    port_mapping = env.summary["node_mapping"]["ports"]
    vessel_mapping = env.summary["node_mapping"]["vessels"]

    # Create a mock gnn state shaper.
    static_code_list, dynamic_code_list = list(port_mapping.values()), list(vessel_mapping.values())
    gnn_state_shaper = GNNStateShaper(
        static_code_list, dynamic_code_list, config.env.param.durations, config.model.feature,
        tick_buffer=config.model.tick_buffer, only_demo=True, max_value=env.configs["total_containers"])
    gnn_state_shaper.compute_static_graph_structure(env)

    # Create an agent_manager.
    config.agent.num_static_nodes = len(static_code_list)
    config.agent.num_dynamic_nodes = len(dynamic_code_list)
    config.agent.algorithm.p2p_adj = gnn_state_shaper.p2p_static_graph
    config.agent.model.port_feature_dim = gnn_state_shaper.get_input_dim("p")
    config.agent.model.vessel_feature_dim = gnn_state_shaper.get_input_dim("v")
    agent = create_gnn_agent(config)
    
    # Learner function for training and testing.
    learner = GNNLearner(
        env, agent, Scheduler(config.main_loop.max_episode), 
        train_freq=config.main_loop.train_freq,
        model_save_freq=config.main_loop.model_save_freq,
        logger=simulation_logger
    )
    learner.learn(config.training)


if __name__ == "__main__":
    from examples.cim.gnn.general import config, distributed_config
    launch(config, distributed_config)
