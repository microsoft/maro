# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
import io
import yaml

from maro.simulator import Env
from maro.rl import ActorProxy, SimpleLearner, AgentMode
from examples.ecr.rl_formulations.common.state_shaper import ECRStateShaper
from maro.utils import Logger, convert_dottable
from examples.ecr.rl_formulations.common.explorer import TwoPhaseLinearExplorer, exploration_config


with io.open("../config.yml", "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    cf = convert_dottable(raw_config)

if cf.rl.modeling == "dqn":
    from examples.ecr.rl_formulations.dqn_agent_manager import DQNAgentManager
    agent_manager_cls = DQNAgentManager
else:  # TODO: enc_gat agent_manager class
    raise ValueError(f"Unsupported RL algorithm: {cf.rl.modeling}")


if __name__ == "__main__":
    env = Env(cf.env.scenario, cf.env.topology, durations=cf.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    state_shaper = ECRStateShaper(**cf.state_shaping)
    explorer = TwoPhaseLinearExplorer(agent_id_list, cf.rl.total_training_episodes, **exploration_config)
    agent_manager = agent_manager_cls(name="ecr_remote_learner", agent_id_list=agent_id_list, mode=AgentMode.TRAIN,
                                      state_shaper=state_shaper, explorer=explorer, seed=cf.rl.seed)

    proxy_params = {"group_name": os.environ['GROUP'],
                    "expected_peers": {"actor_worker": 1},
                    "redis_address": ("localhost", 6379)
                    }
    learner = SimpleLearner(trainable_agents=agent_manager,
                            actor=ActorProxy(proxy_params=proxy_params),
                            logger=Logger("distributed_ecr_learner", auto_timestamp=False),
                            seed=cf.rl.seed)
    time.sleep(15)
    learner.train_test(total_episodes=cf.rl.total_training_episodes)
    learner.dump_models(os.path.join(os.getcwd(), "models"))
