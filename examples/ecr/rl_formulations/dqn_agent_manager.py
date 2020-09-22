# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.nn.functional import smooth_l1_loss
from torch.optim import RMSprop

from maro.rl import AgentManager, Agent, AgentParameters, LearningModel, MLPDecisionLayers, DQN, DQNHyperParams, \
    ExperienceInfoKey

num_actions = 21
model_config = {"hidden_dims": [256, 128, 64], "output_dim": num_actions, "dropout_p": 0.0}
optimizer_config = {"lr": 0.05}
dqn_config = {"num_actions": num_actions, "replace_target_frequency": 5, "tau": 0.1}
training_config = {"min_experiences_to_train": 1024, "samplers": [(lambda d: d[ExperienceInfoKey.TD_ERROR], 128)],
                   "num_steps": 10}


class DQNAgentManager(AgentManager):
    def _assemble_agents(self):
        agent_params = AgentParameters(**training_config)
        for agent_id in self._agent_id_list:
            eval_model = LearningModel(decision_layers=MLPDecisionLayers(name=f'{agent_id}.policy',
                                                                         input_dim=self._state_shaper.dim,
                                                                         **model_config)
                                       )

            algorithm = DQN(model_dict={"eval": eval_model}, optimizer_opt=(RMSprop, optimizer_config),
                            loss_func_dict={"eval": smooth_l1_loss}, hyper_params=DQNHyperParams(**dqn_config))

            self._agent_dict[agent_id] = Agent(name=agent_id, algorithm=algorithm, params=agent_params)
