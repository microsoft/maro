# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import deque
from os import makedirs, system
from os.path import dirname, join, realpath

from torch import nn
from torch.optim import Adam, RMSprop

from maro.rl import (
    ActorCritic, ActorCriticConfig, FullyConnectedBlock, MultiAgentWrapper, OptimOption, SimpleMultiHeadModel
)
from maro.simulator import Env
from maro.utils import Logger, set_seeds

from examples.cim.ac.config import agent_config, training_config
from examples.cim.ac.roll_out import BasicRolloutExecutor


def get_ac_agent():
    actor_net = FullyConnectedBlock(**agent_config["model"]["actor"])
    critic_net = FullyConnectedBlock(**agent_config["model"]["critic"])
    ac_model = SimpleMultiHeadModel(
        {"actor": actor_net, "critic": critic_net}, optim_option=agent_config["optimization"],
    )
    return ActorCritic(ac_model, ActorCriticConfig(**agent_config["hyper_params"]))


# Single-threaded launcher
if __name__ == "__main__":
    set_seeds(1024)  # for reproducibility
    env = Env(**training_config["env"])
    agent = MultiAgentWrapper({name: get_ac_agent() for name in env.agent_idx_list})
    executor = BasicRolloutExecutor(env, agent)
    k, warmup_ep, perf_thresh = training_config["k"], training_config["warmup_ep"], training_config["perf_thresh"]
    perf_history = deque()
    log_path = join(dirname(realpath(__file__)), "logs")
    makedirs(log_path, exist_ok=True)
    logger = Logger("cim_ac", dump_folder=log_path, auto_timestamp=True)
    for ep in range(training_config["max_episode"]):
        exp_by_agent = executor.roll_out(ep)
        logger.info(f"ep-{ep}: {env.metrics}")
        fulfillment = 1 - env.metrics["container_shortage"] / env.metrics["order_requirements"]
        perf_history.append(fulfillment)
        if len(perf_history) > k:
            perf_history.popleft()
        if ep >= warmup_ep and min(perf_history) >= perf_thresh:
            logger.info(f"{k} consecutive fulfillment rates above threshold {perf_thresh}. Training complete")
            break
        # training
        for agent_id, exp in exp_by_agent.items():
            agent[agent_id].train(exp["state"], exp["action"], exp["log_p"], exp["reward"])
