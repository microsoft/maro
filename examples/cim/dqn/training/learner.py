# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time

import numpy as np

from maro.communication import Proxy
from maro.rl import AbsLearner, MultiAgentWrapper, TwoPhaseLinearParameterScheduler, concat
from maro.simulator import Env

from examples.cim.dqn.components import CIMStateShaper, create_dqn_agents


class SimpleLearner(AbsLearner):
    def __init__(
        self, agent, proxy, scheduler, 
        update_trigger=None, inference=False, inference_trigger=None, state_batching_func=np.vstack):
        super().__init__(
            agent, proxy, 
            scheduler=scheduler, 
            update_trigger=update_trigger, 
            inference=inference, 
            inference_trigger=inference_trigger, 
            state_batching_func=state_batching_func
        )

    def run(self):
        for exploration_params in self.scheduler:
            self.agent.set_exploration_params(exploration_params)
            metrics_by_src, exp_by_src = self.collect(self.scheduler.iter, exploration_params=exploration_params)
            for src, metrics in metrics_by_src.items():
                self.logger.info(
                    f"{src}.ep-{self.scheduler.iter} - performance: {metrics}, exploration: {exploration_params}"
                )
            # Store experiences for each agent
            for agent_id, exp in concat(exp_by_src).items():
                exp.update({"loss": [1e8] * len(list(exp.values())[0])})
                self.agent[agent_id].store_experiences(exp)

            for agent in self.agent.agent_dict.values():
                agent.train()

            self.logger.info("Training finished")


def launch(config):
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    config.agent.model.input_dim = CIMStateShaper(**config.env.state_shaping).dim
    config.agent.names = [str(agent_id) for agent_id in env.agent_idx_list]
    agent = MultiAgentWrapper(create_dqn_agents(config.agent))
    scheduler = TwoPhaseLinearParameterScheduler(config.training.max_episode, **config.training.exploration)

    inference = config.multi_process.inference_mode == "remote"
    expected_peers = {"actor": config.multi_process.num_actors}
    if inference:
        expected_peers["rollout_client"] = expected_peers["actor"]
    proxy = Proxy(
        group_name=config.multi_process.group,
        component_type="learner",
        expected_peers=expected_peers,
        redis_address=(config.multi_process.redis.hostname, config.multi_process.redis.port),
        max_retries=15
    )
    
    learner = SimpleLearner(
        agent, proxy, 
        scheduler=scheduler,
        update_trigger=config.multi_process.update_trigger,
        inference=inference,
        inference_trigger=config.multi_process.inference_trigger,
    )

    time.sleep(5)
    learner.run()
    learner.exit()


if __name__ == "__main__":
    from examples.cim.dqn.config import config
    launch(config)
