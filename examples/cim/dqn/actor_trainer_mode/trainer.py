# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl import AgentManagerMode, LearnerActorComponent, concat_experiences_by_agent
from maro.simulator import Env
from maro.utils import convert_dottable

from examples.cim.dqn.components.agent_manager import DQNAgentManager, create_dqn_agents
from examples.cim.dqn.components.config import set_input_dim


def launch(config, distributed_config):
    set_input_dim(config)
    config = convert_dottable(config)
    distributed_config = convert_dottable(distributed_config)
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]

    agent_manager = DQNAgentManager(
        name="distributed_cim_learner",
        mode=AgentManagerMode.TRAIN,
        agent_dict=create_dqn_agents(agent_id_list, config.agents)
    )

    distributed_mode = os.environ.get("MODE", distributed_config.mode)
    if distributed_mode == "seed":
        from maro.rl import SEEDTrainer as trainer_cls
    elif distributed_mode == "simple":
        from maro.rl import Trainer as trainer_cls
    else:
        raise ValueError(f'Supported distributed training modes: "simple", "seed", got {distributed_mode}')

    trainer = trainer_cls(
        agent_manager,
        concat_experiences_by_agent,
        expected_peers={LearnerActorComponent: int(os.environ.get("NUM_ACTORS", distributed_config.num_actors))},
        group_name=os.environ["GROUP"] if "GROUP" in os.environ else distributed_config.group,
        redis_address=(distributed_config.redis.hostname, distributed_config.redis.port),
        max_retries=15
    )
    trainer.launch()


if __name__ == "__main__":
    from examples.cim.dqn.components.config import config, distributed_config
    launch(config=config, distributed_config=distributed_config)
