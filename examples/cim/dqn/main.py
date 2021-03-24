# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from collections import defaultdict
from multiprocessing import Process

from maro.rl import (
    Actor, ActorProxy, DQN, DQNConfig, FullyConnectedBlock, MultiAgentWrapper, OffPolicyLearner,
    SimpleMultiHeadModel, TwoPhaseLinearParameterScheduler
)
from maro.simulator import Env
from maro.utils import set_seeds

from examples.cim.common import CIMTrajectory, common_config
from examples.cim.dqn.config import agent_config, training_config


def get_dqn_agent():
    q_model = SimpleMultiHeadModel(
        FullyConnectedBlock(**agent_config["model"]), optim_option=agent_config["optimization"]
    )
    return DQN(q_model, DQNConfig(**agent_config["hyper_params"]))


class CIMTrajectoryForDQN(CIMTrajectory):
    def on_finish(self):
        exp_by_agent = defaultdict(lambda: defaultdict(list))
        for i in range(len(self.trajectory["state"]) - 1):
            agent_id = list(self.trajectory["state"][i].keys())[0]
            exp = exp_by_agent[agent_id]
            exp["S"].append(self.trajectory["state"][i][agent_id])
            exp["A"].append(self.trajectory["action"][i][agent_id])
            exp["R"].append(self.get_offline_reward(self.trajectory["event"][i]))
            exp["S_"].append(list(self.trajectory["state"][i + 1].values())[0])

        return dict(exp_by_agent)


def cim_dqn_learner():
    env = Env(**training_config["env"])
    agent = MultiAgentWrapper({name: get_dqn_agent() for name in env.agent_idx_list})
    scheduler = TwoPhaseLinearParameterScheduler(training_config["max_episode"], **training_config["exploration"])
    actor = ActorProxy(
        training_config["group"], training_config["num_actors"],
        update_trigger=training_config["learner_update_trigger"]
    )
    learner = OffPolicyLearner(actor, scheduler, agent, **training_config["training"])
    learner.run()


def cim_dqn_actor():
    env = Env(**training_config["env"])
    agent = MultiAgentWrapper({name: get_dqn_agent() for name in env.agent_idx_list})
    actor = Actor(env, agent, CIMTrajectoryForDQN, trajectory_kwargs=common_config)
    actor.as_worker(training_config["group"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--whoami", type=int, choices=[0, 1, 2], default=0,
        help="Identity of this process: 0 - multi-process mode, 1 - learner, 2 - actor"
    )
    
    args = parser.parse_args()
    if args.whoami == 0:
        actor_processes = [Process(target=cim_dqn_actor) for _ in range(training_config["num_actors"])]
        learner_process = Process(target=cim_dqn_learner)

        for i, actor_process in enumerate(actor_processes):
            set_seeds(i)  # this is to ensure that the actors explore differently.
            actor_process.start()

        learner_process.start()

        for actor_process in actor_processes:
            actor_process.join()

        learner_process.join()
    elif args.whoami == 1:
        cim_dqn_learner()
    elif args.whoami == 2:
        cim_dqn_actor()
