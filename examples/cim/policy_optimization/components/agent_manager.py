# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

import numpy as np
import torch.nn as nn
from torch.optim import Adam, RMSprop

from maro.rl import (
    AbsAgentManager, ActorCritic, ActorCriticConfig, FullyConnectedBlock, OptimOption,
    PolicyGradient, SimpleMultiHeadModel
)
from maro.utils import set_seeds

from examples.cim.policy_optimization.components.state_shaper import CIMStateShaper
from examples.cim.policy_optimization.components.action_shaper import CIMActionShaper
from examples.cim.policy_optimization.components.experience_shaper import TruncatedExperienceShaper


def create_po_agents(agent_id_list, config):
    input_dim, num_actions = config.input_dim, config.num_actions
    set_seeds(config.seed)
    agent_dict = {}
    for agent_id in agent_id_list:
        actor_net = FullyConnectedBlock(
            input_dim=input_dim,
            output_dim=num_actions,
            activation=nn.Tanh,
            is_head=True,
            **config.actor_model
        )

        if config.type == "actor_critic":
            critic_net = FullyConnectedBlock(
                input_dim=config.input_dim,
                output_dim=1,
                activation=nn.LeakyReLU,
                is_head=True,
                **config.critic_model
            )

            hyper_params = config.actor_critic_hyper_parameters
            hyper_params.update({"reward_discount": config.reward_discount})
            learning_model = SimpleMultiHeadModel(
                {"actor": actor_net, "critic": critic_net}, 
                optim_option={
                    "actor": OptimOption(optim_cls=Adam, optim_params=config.actor_optimizer),
                    "critic": OptimOption(optim_cls=RMSprop, optim_params=config.critic_optimizer)
                }
            )
            agent_dict[agent_id] = ActorCritic(
                agent_id, learning_model, ActorCriticConfig(critic_loss_func=nn.SmoothL1Loss(), **hyper_params)
            )
        else:
            learning_model = SimpleMultiHeadModel(
                actor_net, 
                optim_option=OptimOption(optim_cls=Adam, optim_params=config.actor_optimizer)
            )
            agent_dict[agent_id] = PolicyGradient(agent_id, learning_model, config.reward_discount)

    return agent_dict


class POAgentManager(AbsAgentManager):
    def __init__(
        self,
        agent,
        state_shaper: CIMStateShaper,
        action_shaper: CIMActionShaper,
        experience_shaper: TruncatedExperienceShaper
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
        action, log_p = self.agent[agent_id].choose_action(model_state)
        self._trajectory["state"].append(model_state)
        self._trajectory["agent_id"].append(agent_id)
        self._trajectory["event"].append(decision_event)
        self._trajectory["action"].append(action)
        self._trajectory["log_action_prob"].append(log_p)
        return self._action_shaper(action, decision_event, snapshot_list)

    def train(self, experiences_by_agent: dict):
        for agent_id, exp in experiences_by_agent.items():
            if not isinstance(exp, list):
                exp = [exp]
            for trajectory in exp:
                self.agent[agent_id].train(
                    trajectory["state"],
                    trajectory["action"],
                    trajectory["log_action_prob"],
                    trajectory["reward"]
                )

    def on_env_feedback(self, metrics):
        self._trajectory["metrics"].append(metrics)

    def post_process(self, snapshot_list):
        experiences = self._experience_shaper(self._trajectory, snapshot_list)
        self._trajectory.clear()
        self._state_shaper.reset()
        self._action_shaper.reset()
        self._experience_shaper.reset()
        return experiences
