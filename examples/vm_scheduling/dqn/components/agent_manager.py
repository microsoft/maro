# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

import torch.nn as nn
from torch.optim import RMSprop, lr_scheduler

from maro.rl import (
    AbsAgentManager, DQNConfig, FullyConnectedBlock, OptimOption, SimpleMultiHeadModel,
    SimpleStore, AbsAgent
)
from maro.utils import set_seeds

from examples.vm_scheduling.dqn.components.state_shaper import VMStateShaper
from examples.vm_scheduling.dqn.components.action_shaper import VMActionShaper
from examples.vm_scheduling.dqn.components.experience_shaper import TruncatedExperienceShaper
from examples.vm_scheduling.dqn.vm_rl import VMDQN
from examples.vm_scheduling.dqn.vm_rl.models.combine_net import CombineNet
from examples.vm_scheduling.dqn.vm_rl.models.split_net import SplitNet


def create_dqn_agents(agent_name, config):
    set_seeds(config.seed)
    q_net = CombineNet(
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
    agent = VMDQN(
        agent_name, learning_model, DQNConfig(**config.hyper_params, loss_cls=nn.SmoothL1Loss)
    )

    return agent


class DQNAgentManager(AbsAgentManager):
    def __init__(
        self,
        agent,
        state_shaper: VMStateShaper = None,
        action_shaper: VMActionShaper = None,
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

    def choose_action(self, decision_event, env):
        model_state, legal_action = self._state_shaper(decision_event, env)
        action = self.agent.choose_action(model_state, legal_action)
        self._trajectory["state"].append(model_state)
        self._trajectory["event"].append(decision_event)
        self._trajectory["action"].append(action)
        self._trajectory["legal_action"].append(legal_action)
        return self._action_shaper(action, decision_event)

    def train(self, experiences_by_agent):
        # store experiences for each agent
        for agent_id, exp in experiences_by_agent.items():
            exp.update({"loss": [1e8] * len(list(exp.values())[0])})
            self.agent.store_experiences(exp)

        self.agent.train()

    def on_env_feedback(self, metrics):
        self._trajectory["metrics"].append(metrics)

    def post_process(self):
        experiences = self._experience_shaper(self._trajectory)
        self._trajectory.clear()
        self._state_shaper.reset()
        self._action_shaper.reset()
        self._experience_shaper.reset()
        return experiences

    def load_models(self, agent_model_dict):
        """Load models from memory for each agent."""
        if isinstance(self.agent, AbsAgent):
            self.agent.load_model(agent_model_dict)
        else:
            for agent_id, models in agent_model_dict.items():
                self.agent[agent_id].load_model(models)
