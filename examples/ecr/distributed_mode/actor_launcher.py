# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import io
import yaml

import numpy as np

from maro.simulator import Env
from maro.rl import AgentMode, SimpleActor, ActorWorker, KStepRewardShaper
from maro.utils import convert_dottable
from examples.ecr.rl_formulations.common.state_shaper import ECRStateShaper
from examples.ecr.rl_formulations.common.action_shaper import ECRActionShaper
from examples.ecr.rl_formulations.common.reward_shaper import ECRRewardShaper
from examples.ecr.rl_formulations.common.explorer import TwoPhaseLinearExplorer, exploration_config


with io.open("../config.yml", "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    cf = convert_dottable(raw_config)

if cf.rl.modeling == "dqn":
    from examples.ecr.rl_formulations.dqn_agent_manager import DQNAgentManager, num_actions
    agent_manager_cls = DQNAgentManager
    action_space = list(np.linspace(-1.0, 1.0, num_actions))
else:  # TODO: enc_gat agent_manager class
    raise ValueError(f"Unsupported RL algorithm: {cf.rl.modeling}")


if __name__ == "__main__":
    env = Env(cf.env.scenario, cf.env.topology, durations=cf.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    state_shaper = ECRStateShaper(**cf.state_shaping)
    action_shaper = ECRActionShaper(action_space=action_space)
    if cf.reward_shaping.type == "truncated":
        reward_shaper = ECRRewardShaper(agent_id_list=agent_id_list, **cf.reward_shaping.truncated)
    else:
        reward_shaper = KStepRewardShaper(reward_func=lambda mt: mt["perf"], **cf.reward_shaping.k_step)
    explorer = TwoPhaseLinearExplorer(agent_id_list, cf.rl.total_training_episodes, **exploration_config)
    agent_manager = agent_manager_cls(name="ecr_remote_actor",
                                      agent_id_list=agent_id_list,
                                      mode=AgentMode.INFERENCE,
                                      state_shaper=state_shaper,
                                      action_shaper=action_shaper,
                                      reward_shaper=reward_shaper,
                                      explorer=explorer,
                                      seed=cf.rl.seed)

    proxy_params = {"group_name": os.environ["GROUP"],
                    "expected_peers": {"actor_proxy": 1},
                    "redis_address": ("localhost", 6379)
                    }
    actor_worker = ActorWorker(local_actor=SimpleActor(env=env, inference_agents=agent_manager),
                               proxy_params=proxy_params)
    actor_worker.launch()
