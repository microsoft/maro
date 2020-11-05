# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from components.agent_manager import DQNAgentManager, create_dqn_agents
from components.config import set_input_dim

from maro.rl import (
    ActorProxy, AgentManagerMode, MaxDeltaEarlyStoppingChecker, SimpleLearner, TwoPhaseLinearExplorer,
    concat_experiences_by_agent
)
from maro.simulator import Env
from maro.utils import Logger, convert_dottable


def launch(config):
    set_input_dim(config)
    config = convert_dottable(config)
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]

    agent_manager = DQNAgentManager(
        name="distributed_cim_learner",
        mode=AgentManagerMode.TRAIN,
        agent_dict=create_dqn_agents(agent_id_list, config.agents),
    )

    proxy_params = {
        "group_name": os.environ["GROUP"],
        "expected_peers": {"actor": int(os.environ["NUM_ACTORS"])},
        "redis_address": ("localhost", 6379)
    }

    early_stopping_checker = MaxDeltaEarlyStoppingChecker(
        last_k=config.general.early_stopping.last_k,
        threshold=config.general.early_stopping.threshold
    )

    learner = SimpleLearner(
        trainable_agents=agent_manager,
        actor=ActorProxy(proxy_params=proxy_params, experience_collecting_func=concat_experiences_by_agent),
        explorer=TwoPhaseLinearExplorer(**config.exploration),
        logger=Logger("distributed_cim_learner", auto_timestamp=False)
    )
    learner.train(
        max_episode=config.general.max_episode,
        early_stopping_checker=early_stopping_checker,
        warmup_ep=config.general.early_stopping.warmup_ep,
        early_stopping_metric_func=lambda x: 1 - x["container_shortage"] / x["order_requirements"],
    )
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))
    learner.exit()


if __name__ == "__main__":
    from components.config import config
    launch(config)
