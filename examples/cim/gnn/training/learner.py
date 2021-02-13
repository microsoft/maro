import datetime
import os
import time

from maro.communication import Proxy
from maro.rl import AbsLearner
from maro.simulator import Env
from maro.utils import Logger

from examples.cim.gnn.components import GNNStateShaper, create_gnn_agent
from examples.cim.gnn.general import simulation_logger
from examples.cim.gnn.training.utils import (
    batch_states, combine, decision_cnt_analysis, load_config, save_code, save_config
)


class SimpleLearner(AbsLearner):
    """Learner class for the training pipeline and the specialized logging in GNN solution for CIM problem."""
    def __init__(
        self, group_name, num_actors, agent, max_ep, 
        inference=False, state_batching_func=batch_states, train_freq=1, model_save_freq=1, log_dir=os.getcwd()
    ):
        super().__init__(
            group_name, num_actors, agent, inference=inference, state_batching_func=state_batching_func
        )
        self._max_ep = max_ep
        self._train_freq = train_freq
        self._model_save_freq = model_save_freq
        self._log_pth = os.path.join(log_dir, "model")
        os.makedirs(self._log_pth, exist_ok=True)

    def run(self):
        rollout_time = 0
        training_time = 0
        for ep in range(self._max_ep):
            rollout_start = time.time()
            metrics_by_src, details_by_src = self.collect(ep)
            exp_by_src = {src: exp for src, (exp, logs) in details_by_src.items()}
            rollout_time += time.time() - rollout_start
            for src, metrics in metrics_by_src.items():
                self.logger.info(f"{src}.ep-{ep} - performance: {metrics}")
            train_start = time.time()
            self.agent.store_experiences(combine(exp_by_src))
            if ep % self._train_freq == self._train_freq - 1:
                self.logger.info("training")
                self.agent.train()
            training_time += time.time() - train_start
            if self._log_pth is not None and (ep + 1) % self._model_save_freq == 0:
                self.agent.dump_model_to_file(os.path.join(self._log_pth, str(ep + 1)))

            self.logger.debug(f"rollout time: {int(rollout_time)}")
            self.logger.debug(f"training time: {int(training_time)}")


def launch(config):
    # Create a demo environment to retrieve environment information.
    simulation_logger.info("Getting experience quantity estimates for each (port, vessel) pair...")
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    exp_per_ep = decision_cnt_analysis(env, pv=True, buffer_size=8)
    config.agent.exp_per_ep = {
        agent_id: cnt * config.multi_process.num_actors * config.training.train_freq
        for agent_id, cnt in exp_per_ep.items()
    }
    simulation_logger.info(config.agent.exp_per_ep)

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

    config.agent.num_static_nodes = len(static_code_list)
    config.agent.num_dynamic_nodes = len(dynamic_code_list)
    config.agent.hyper_params.p2p_adj = state_shaper.p2p_static_graph
    config.agent.model.port_feature_dim = state_shaper.get_input_dim("p")
    config.agent.model.vessel_feature_dim = state_shaper.get_input_dim("v")

    learner = SimpleLearner(
        config.multi_process.group,
        config.multi_process.num_actors,
        create_gnn_agent(config.agent),
        config.training.max_episode,
        inference=config.multi_process.inference_mode=="remote",
        log_dir=config.log.path
    )

    time.sleep(5)
    learner.run()
    learner.exit()


if __name__ == "__main__":
    from examples.cim.gnn.general import config
    launch(config)
