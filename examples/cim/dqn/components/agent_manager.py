# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.nn.functional import smooth_l1_loss
from torch.optim import RMSprop

from .agent import CIMAgent
from .config import config
from maro.rl import AbsAgentManager, LearningModel, MLPDecisionLayers, DQN, DQNHyperParams, ColumnBasedStore
from maro.utils import set_seeds


class DQNAgentManager(AbsAgentManager):
    def _assemble(self, agent_dict):
        set_seeds(config.agents.seed)
        num_actions = config.agents.algorithm.num_actions
        for agent_id in self._agent_id_list:
            eval_model = LearningModel(
                decision_layers=MLPDecisionLayers(
                                    name=f'{agent_id}.policy', input_dim=self._state_shaper.dim,
                                    output_dim=num_actions, **config.agents.algorithm.model
                                )
            )

            algorithm = DQN(
                eval_model=eval_model,
                optimizer_cls=RMSprop,
                optimizer_params=config.agents.algorithm.optimizer,
                loss_func=smooth_l1_loss,
                hyper_params=DQNHyperParams(
                    **config.agents.algorithm.hyper_parameters,
                    num_actions=num_actions
                )
            )

            experience_pool = ColumnBasedStore(**config.agents.experience_pool)
            agent_dict[agent_id] = CIMAgent(name=agent_id, algorithm=algorithm, experience_pool=experience_pool,
                                            **config.agents.training_loop_parameters)

    def choose_action(self, decision_event, snapshot_list):
        self._assert_inference_mode()
        agent_id, model_state = self._state_shaper(decision_event, snapshot_list)
        model_action = self._agent_dict[agent_id].choose_action(
            model_state, self._explorer.epsilon[agent_id] if self._explorer else None)

        self._transition_cache = {"state": model_state,
                                  "action": model_action,
                                  "reward": None,
                                  "agent_id": agent_id,
                                  "event": decision_event}
        return self._action_shaper(model_action, decision_event, snapshot_list)

    def train(self, experiences_by_agent, performance=None):
        self._assert_train_mode()

        # store experiences for each agent
        for agent_id, exp in experiences_by_agent.items():
            exp.update({"loss": [1e8] * len(exp[next(iter(exp))])})
            self._agent_dict[agent_id].store_experiences(exp)

        for agent in self._agent_dict.values():
            agent.train()

        # update exploration rates
        if self._explorer is not None:
            self._explorer.update(performance)
