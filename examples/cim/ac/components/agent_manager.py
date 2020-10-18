# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.nn.functional import smooth_l1_loss
from torch.optim import Adam, RMSprop

from .agent import CIMAgent
from .config import config
from maro.rl import AbsAgentManager, LearningModel, MLPDecisionLayers, ActorCritic, ActorCriticHyperParameters
from maro.utils import set_seeds


class ACAgentManager(AbsAgentManager):
    def _assemble(self, agent_dict):
        set_seeds(config.agents.seed)
        num_actions = config.agents.algorithm.num_actions
        for agent_id in self._agent_id_list:
            policy_model = LearningModel(
                decision_layers=MLPDecisionLayers(
                    name=f'{agent_id}.policy',
                    input_dim=self._state_shaper.dim,
                    output_dim=num_actions,
                    **config.agents.algorithm.policy_model,
                    softmax=True
                )
            )

            value_model = LearningModel(
                decision_layers=MLPDecisionLayers(
                    name=f'{agent_id}.policy',
                    input_dim=self._state_shaper.dim,
                    output_dim=num_actions,
                    **config.agents.algorithm.value_model
                )
            )

            algorithm = ActorCritic(
                policy_model=policy_model,
                value_model=value_model,
                value_loss_func=smooth_l1_loss,
                policy_optimizer_cls=Adam,
                policy_optimizer_params=config.agents.algorithm.policy_optimizer,
                value_optimizer_cls=RMSprop,
                value_optimizer_params=config.agents.algorithm.value_optimizer,
                hyper_params=ActorCriticHyperParameters(
                                num_actions=num_actions,
                                **config.agents.algorithm.hyper_parameters,
                            )
            )

            agent_dict[agent_id] = CIMAgent(name=agent_id, algorithm=algorithm)

    def choose_action(self, decision_event, snapshot_list):
        self._assert_inference_mode()
        agent_id, model_state = self._state_shaper(decision_event, snapshot_list)
        model_action = self._agent_dict[agent_id].choose_action(
            model_state, self._explorer.epsilon[agent_id] if self._explorer else None
        )

        self._transition_cache = {"state": model_state,
                                  "action": model_action,
                                  "reward": None,
                                  "agent_id": agent_id,
                                  "event": decision_event}
        return self._action_shaper(model_action, decision_event, snapshot_list)

    def train(self, experiences_by_agent: dict):
        for agent_id, experiences in experiences_by_agent.items():
            self._agent_dict[agent_id].train(experiences["states"], experiences["actions"], experiences["rewards"])

    def store_experiences(self, experiences):
        pass
