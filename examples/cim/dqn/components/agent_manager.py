# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.nn.functional import smooth_l1_loss
from torch.optim import RMSprop

from .agent import CIMAgent
from maro.rl import AgentMode, SimpleAgentManager, LearningModel, DecisionLayers, DQN, DQNHyperParams, ColumnBasedStore
from maro.utils import set_seeds


def create_dqn_agents(agent_id_list, mode, config):
    if mode in {AgentMode.TRAIN, AgentMode.TRAIN_INFERENCE}:
        return {agent_id: DQN(
                            eval_model=None,
                            optimizer_cls=None,
                            optimizer_params=None,
                            loss_func=None,
                            hyper_params=None
                            )
                for agent_id in agent_id_list}

    set_seeds(config.seed)
    num_actions = config.algorithm.num_actions
    agent_dict = {}
    for agent_id in agent_id_list:
        eval_model = LearningModel(
            decision_layers=DecisionLayers(
                name=f'{agent_id}.policy', input_dim=config.algorithm.input_dim,
                output_dim=num_actions, **config.algorithm.model
            )
        )

        algorithm = DQN(
            eval_model=eval_model,
            optimizer_cls=RMSprop,
            optimizer_params=config.algorithm.optimizer,
            loss_func=smooth_l1_loss,
            hyper_params=DQNHyperParams(
                **config.algorithm.hyper_parameters,
                num_actions=num_actions
            )
        )

        experience_pool = ColumnBasedStore(**config.experience_pool)
        agent_dict[agent_id] = CIMAgent(name=agent_id, mode=mode, algorithm=algorithm, experience_pool=experience_pool,
                                        **config.training_loop_parameters)


class DQNAgentManager(SimpleAgentManager):
    def train(self, experiences_by_agent, performance=None):
        self._assert_train_mode()

        # store experiences for each agent
        for agent_id, exp in experiences_by_agent.items():
            exp.update({"loss": [1e8] * len(exp[next(iter(exp))])})
            self.agent_dict[agent_id].store_experiences(exp)

        for agent in self.agent_dict.values():
            agent.train()

        # update exploration rates
        if self._explorer is not None:
            self._explorer.update(performance)
