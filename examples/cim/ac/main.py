# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl import (
    Actor, ActorCritic, ActorCriticConfig, FullyConnectedBlock, MultiAgentWrapper, SimpleMultiHeadModel,
    Scheduler, OnPolicyLearner
)
from maro.simulator import Env
from maro.utils import set_seeds

from examples.cim.ac.config import agent_config, training_config
from examples.cim.common import CIMTrajectory, common_config


def get_ac_agent():
    actor_net = FullyConnectedBlock(**agent_config["model"]["actor"])
    critic_net = FullyConnectedBlock(**agent_config["model"]["critic"])
    ac_model = SimpleMultiHeadModel(
        {"actor": actor_net, "critic": critic_net}, optim_option=agent_config["optimization"],
    )
    return ActorCritic(ac_model, ActorCriticConfig(**agent_config["hyper_params"]))


class CIMTrajectoryForAC(CIMTrajectory):
    def on_finish(self):
        training_data = {}
        for event, state, action in zip(self.trajectory["event"], self.trajectory["state"], self.trajectory["action"]):
            agent_id = list(state.keys())[0]
            data = training_data.setdefault(agent_id, {"args": [[] for _ in range(4)]})
            data["args"][0].append(state[agent_id])  # state
            data["args"][1].append(action[agent_id][0])  # action
            data["args"][2].append(action[agent_id][1])  # log_p
            data["args"][3].append(self.get_offline_reward(event))  # reward

        for agent_id in training_data:
            training_data[agent_id]["args"] = [
                np.asarray(vals, dtype=np.float32 if i == 3 else None)
                for i, vals in enumerate(training_data[agent_id]["args"])
            ]

        return training_data


# Single-threaded launcher
if __name__ == "__main__":
    set_seeds(1024)  # for reproducibility
    env = Env(**training_config["env"])
    agent = MultiAgentWrapper({name: get_ac_agent() for name in env.agent_idx_list})
    actor = Actor(env, agent, CIMTrajectoryForAC, trajectory_kwargs=common_config)  # local actor
    learner = OnPolicyLearner(actor, training_config["max_episode"])
    learner.run()
