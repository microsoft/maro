# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.nn.functional import smooth_l1_loss
from torch.optim import RMSprop

from maro.rl import DQN, AbsAgentManager, ColumnBasedStore, DQNHyperParams, LearningModel, MLPDecisionLayers
from maro.utils import set_seeds

from .agent import CIMAgent
from .config import config


class DQNAgentManager(AbsAgentManager):
    def _assemble(self, agent_dict):
        set_seeds(config.agents.seed)
        num_actions = config.agents.algorithm.num_actions
        for agent_id in self._agent_id_list:
            eval_model = LearningModel(decision_layers=MLPDecisionLayers(name=f'{agent_id}.policy',
                                                                         input_dim=self._state_shaper.dim,
                                                                         output_dim=num_actions,
                                                                         **config.agents.algorithm.model)
                                       )

            algorithm = DQN(model_dict={"eval": eval_model},
                            optimizer_opt=(RMSprop, config.agents.algorithm.optimizer),
                            loss_func_dict={"eval": smooth_l1_loss},
                            hyper_params=DQNHyperParams(**config.agents.algorithm.hyper_parameters,
                                                        num_actions=num_actions))

            experience_pool = ColumnBasedStore(**config.agents.experience_pool)
            agent_dict[agent_id] = CIMAgent(name=agent_id, algorithm=algorithm, experience_pool=experience_pool,
                                            **config.agents.training_loop_parameters)

    def store_experiences(self, experiences):
        for agent_id, exp in experiences.items():
            exp.update({"loss": [1e8] * len(exp[next(iter(exp))])})
            self._agent_dict[agent_id].store_experiences(exp)
