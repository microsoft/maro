# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

import torch.nn as nn
from torch.optim import RMSprop, lr_scheduler

from maro.rl import (
    AbsAgentManager, DQN, DQNConfig, FullyConnectedBlock, OptimOption, SimpleMultiHeadModel,
    SimpleStore
)
from maro.utils import set_seeds

from examples.cim.dqn.components.state_shaper import CIMStateShaper
from examples.cim.dqn.components.action_shaper import CIMActionShaper
from examples.cim.dqn.components.experience_shaper import TruncatedExperienceShaper


def create_dqn_agents(agent_names, config):
    set_seeds(config.seed)
    agent_dict = {}
    for name in agent_names:
        q_net = FullyConnectedBlock(
            activation=nn.LeakyReLU,
            is_head=True,
            **config.model
        )            
        learning_model = SimpleMultiHeadModel(
            q_net,
            optim_option=OptimOption(
                optim_cls=RMSprop, 
                optim_params=config.optimizer,
                scheduler_cls=lr_scheduler.StepLR,
                scheduler_params=config.scheduler    
            )
        )
        agent_dict[name] = DQN(
            name, learning_model, DQNConfig(**config.hyper_params, loss_cls=nn.SmoothL1Loss)
        )

    return agent_dict


class DQNAgentManager(AbsAgentManager):
    def __init__(
        self,
        agent,
        state_shaper: CIMStateShaper = None,
        action_shaper: CIMActionShaper = None,
        experience_shaper: TruncatedExperienceShaper = None
    ):
        super().__init__(
            agent,
            state_shaper=state_shaper,
            action_shaper=action_shaper,
            experience_shaper=experience_shaper
        )
        # Data structure to temporarily store the trajectory
        self._trajectory = defaultdict(list)

    def choose_action(self, decision_event, snapshot_list):
        agent_id, model_state = self._state_shaper(decision_event, snapshot_list)
        action = self.agent[agent_id].choose_action(model_state)
        self._trajectory["state"].append(model_state)
        self._trajectory["agent_id"].append(agent_id)
        self._trajectory["event"].append(decision_event)
        self._trajectory["action"].append(action)
        return self._action_shaper(action, decision_event, snapshot_list)

    def train(self, experiences_by_agent):
        # store experiences for each agent
        for agent_id, exp in experiences_by_agent.items():
            exp.update({"loss": [1e8] * len(list(exp.values())[0])})
            self.agent[agent_id].store_experiences(exp)

        for agent in self.agent.values():
            agent.train()

    def on_env_feedback(self, metrics):
        self._trajectory["metrics"].append(metrics)

    def post_process(self, snapshot_list):
        experiences = self._experience_shaper(self._trajectory, snapshot_list)
        self._trajectory.clear()
        self._state_shaper.reset()
        self._action_shaper.reset()
        self._experience_shaper.reset()
        return experiences
