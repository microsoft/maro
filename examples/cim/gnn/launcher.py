import datetime
import os

from maro.simulator import Env
from maro.utils import Logger

from .actor import ParallelActor
from .agent_manager import SimpleAgentManger
from .learner import GNNLearner
from .state_shaper import GNNStateShaper
from .utils import decision_cnt_analysis, load_config, return_scaler, save_code, save_config

if __name__ == "__main__":
    config_pth = "examples/cim/gnn/config.yml"
    config = load_config(config_pth)

    # Generate log path.
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    time_str = datetime.datetime.now().strftime("%H%M%S.%f")
    subfolder_name = f"{config.env.param.topology}_{time_str}"

    # Log path.
    config.log.path = os.path.join(config.log.path, date_str, subfolder_name)
    if not os.path.exists(config.log.path):
        os.makedirs(config.log.path)

    simulation_logger = Logger(tag="simulation", dump_folder=config.log.path, dump_mode="w", auto_timestamp=False)

    # Create a demo environment to retrieve environment information.
    simulation_logger.info("Approximating the experience quantity of each agent...")
    demo_env = Env(**config.env.param)
    config.env.exp_per_ep = decision_cnt_analysis(demo_env, pv=True, buffer_size=8)
    simulation_logger.info(config.env.exp_per_ep)

    # Add some buffer to prevent overlapping.
    config.env.return_scaler, tot_order_amount = return_scaler(
        demo_env, tick=config.env.param.durations, gamma=config.training.gamma)
    simulation_logger.info(f"Return value will be scaled down by the factor {config.env.return_scaler}")

    save_config(config, os.path.join(config.log.path, "config.yml"))
    save_code("examples/cim/gnn", config.log.path)

    port_mapping = demo_env.summary["node_mapping"]["ports"]
    vessel_mapping = demo_env.summary["node_mapping"]["vessels"]

    # Create a mock gnn_state_shaper.
    static_code_list, dynamic_code_list = list(port_mapping.values()), list(vessel_mapping.values())
    gnn_state_shaper = GNNStateShaper(
        static_code_list, dynamic_code_list, config.env.param.durations, config.model.feature,
        tick_buffer=config.model.tick_buffer, only_demo=True, max_value=demo_env.configs["total_containers"])
    gnn_state_shaper.compute_static_graph_structure(demo_env)

    # Create and assemble agent_manager.
    agent_id_list = list(config.env.exp_per_ep.keys())
    training_logger = Logger(tag="training", dump_folder=config.log.path, dump_mode="w", auto_timestamp=False)
    agent_manager = SimpleAgentManger(
        "CIM-GNN-manager", agent_id_list, static_code_list, dynamic_code_list, demo_env, gnn_state_shaper,
        training_logger)
    agent_manager.assemble(config)

    # Create the rollout actor to collect experience.
    actor = ParallelActor(config, demo_env, gnn_state_shaper, agent_manager, logger=simulation_logger)

    # Learner function for training and testing.
    learner = GNNLearner(actor, agent_manager, logger=simulation_logger)
    learner.train(config.training)

    # Cancel all the child process used for rollout.
    actor.exit()
