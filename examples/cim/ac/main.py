# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl import (
    Actor, ActorCritic, ActorCriticConfig, FullyConnectedBlock, MultiAgentWrapper, SimpleMultiHeadModel,
    OnPolicyLearner
)
from maro.simulator import Env
from maro.utils import set_seeds

from examples.cim.common import CIMEnvWrapper
from examples.cim.common_config import common_config
from examples.cim.ac.config import config


def get_ac_agent():
    actor_net = FullyConnectedBlock(**config["agent"]["model"]["actor"])
    critic_net = FullyConnectedBlock(**config["agent"]["model"]["critic"])
    ac_model = SimpleMultiHeadModel(
        {"actor": actor_net, "critic": critic_net}, optim_option=config["agent"]["optimization"],
    )
    return ActorCritic(ac_model, ActorCriticConfig(**config["agent"]["hyper_params"]))


# Single-threaded launcher
if __name__ == "__main__":
    set_seeds(1024)  # for reproducibility
    env = Env(**config["training"]["env"])
    agent = MultiAgentWrapper({name: get_ac_agent() for name in env.agent_idx_list})
    learner = OnPolicyLearner(CIMEnvWrapper(env, **common_config), agent, config["training"]["max_episode"])
    learner.run()
