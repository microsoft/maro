import datetime
import os
import time

from maro.rl import Scheduler
from maro.simulator import Env
from maro.utils import Logger

from examples.cim.gnn.components import DiscreteActionShaper, GNNExperienceShaper, GNNStateShaper, create_gnn_agent
from examples.cim.gnn.general import simulation_logger, training_logger
from examples.cim.gnn.training.actor import BasicRolloutExecutor
from examples.cim.gnn.training.utils import decision_cnt_analysis, load_config, return_scaler, save_code, save_config


def launch(config):
    # Create a demo environment to retrieve environment information.
    simulation_logger.info("Getting experience quantity estimates for each (port, vessel) pair...")
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    exp_per_ep = decision_cnt_analysis(env, pv=True, buffer_size=8)
    config.agent.exp_per_ep = {
        agent_id: cnt * config.training.train_freq for agent_id, cnt in exp_per_ep.items()
    }
    simulation_logger.info(config.agent.exp_per_ep)

    # Add some buffer to prevent overlapping.
    config.env.return_scaler, _ = return_scaler(
        env, tick=config.env.durations, gamma=config.agent.hyper_params.reward_discount
    )
    simulation_logger.info(f"Return values will be scaled down by a factor of {config.env.return_scaler}")

    save_config(config, os.path.join(config.log.path, "config.yml"))
    # save_code("examples/cim/gnn", config.log.path)

    static_code_list = list(env.summary["node_mapping"]["ports"].values())
    dynamic_code_list = list(env.summary["node_mapping"]["vessels"].values())

    # Create a mock gnn state shaper.
    state_shaper = GNNStateShaper(
        static_code_list, dynamic_code_list, config.env.durations, config.agent.model.feature,
        sequence_buffer_size=config.agent.model.sequence_buffer_size, max_value=env.configs["total_containers"]
    )
    state_shaper.compute_static_graph_structure(env)
    action_shaper = DiscreteActionShaper(config.agent.model.action_dim)
    experience_shaper = GNNExperienceShaper(
        static_code_list, dynamic_code_list, config.env.durations, state_shaper,
        scale_factor=config.env.return_scaler, time_slot=config.agent.hyper_params.td_steps,
        discount_factor=config.agent.hyper_params.reward_discount
    )

    config.agent.num_static_nodes = len(static_code_list)
    config.agent.num_dynamic_nodes = len(dynamic_code_list)
    config.agent.hyper_params.p2p_adj = state_shaper.p2p_static_graph
    config.agent.model.port_feature_dim = state_shaper.get_input_dim("p")
    config.agent.model.vessel_feature_dim = state_shaper.get_input_dim("v")
    
    agent = create_gnn_agent(config.agent)
    executor = BasicRolloutExecutor(env, agent, state_shaper, action_shaper, experience_shaper)
    rollout_time = training_time = 0
    train_freq, model_save_freq = config.training.train_freq, config.training.model_save_freq 
    log_dir = os.path.join(config.log.path, "models")
    os.makedirs(log_dir, exist_ok=True)
    for ep in range(config.training.max_episode):
        tick = time.time()
        exp, _ = executor.roll_out(ep)
        rollout_time += time.time() - tick
        simulation_logger.info(f"ep {ep} - metrics: {env.metrics}")
        agent.store_experiences(exp)
        if ep % train_freq == train_freq - 1:
            simulation_logger.info("training")
            tick = time.time()
            agent.train()
            training_time += time.time() - tick
        if (ep + 1) % model_save_freq == 0:
            agent.dump_model_to_file(os.path.join(log_dir, str(ep)))

        simulation_logger.debug(f"rollout time: {int(rollout_time)}")
        simulation_logger.debug(f"training time: {int(training_time)}")


if __name__ == "__main__":
    from examples.cim.gnn.general import config
    # multi-process mode
    if config.multi_process.enable:
        learner_path = f"{os.path.split(os.path.realpath(__file__))[0]}/training/learner.py &"
        actor_path = f"{os.path.split(os.path.realpath(__file__))[0]}/training/actor.py &"

        # Launch the actor processes
        for _ in range(config.multi_process.num_actors):
            os.system(f"python {actor_path}")

        # Launch the learner process
        os.system(f"python {learner_path}")
    else:
        launch(config)
