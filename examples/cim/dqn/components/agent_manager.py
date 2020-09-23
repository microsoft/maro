# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import yaml

from torch.nn.functional import smooth_l1_loss
from torch.optim import RMSprop

from maro.rl import AbsAgentManager, LearningModel, MLPDecisionLayers, DQN, DQNHyperParams, ColumnBasedStore
from maro.utils import convert_dottable, set_seeds
from .agent import CIMAgent


with io.open("config.yml", "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)
    config = config.agents


class DQNAgentManager(AbsAgentManager):
    def _assemble(self, agent_dict):
        set_seeds(config.seed)
        num_actions = config.algorithm.num_actions
        for agent_id in self._agent_id_list:
            eval_model = LearningModel(decision_layers=MLPDecisionLayers(name=f'{agent_id}.policy',
                                                                         input_dim=self._state_shaper.dim,
                                                                         output_dim=num_actions,
                                                                         **config.algorithm.model)
                                       )

            algorithm = DQN(model_dict={"eval": eval_model},
                            optimizer_opt=(RMSprop, config.algorithm.optimizer),
                            loss_func_dict={"eval": smooth_l1_loss},
                            hyper_params=DQNHyperParams(**config.algorithm.hyper_parameters,
                                                        num_actions=num_actions))

            experience_pool = ColumnBasedStore(**config.experience_pool)
            agent_dict[agent_id] = CIMAgent(name=agent_id, algorithm=algorithm, experience_pool=experience_pool,
                                            **config.training_loop_parameters)

    def store_experiences(self, experiences):
        for agent_id, exp in experiences.items():
            exp.update({"loss": [1e8] * len(exp[next(iter(exp))])})
            self._agent_dict[agent_id].store_experiences(exp)
