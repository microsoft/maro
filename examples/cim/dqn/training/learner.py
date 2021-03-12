# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl import AbsLearner, SimpleStore, concat
from maro.utils import DummyLogger


class BasicLearner(AbsLearner):
    def __init__(
        self, group_name, num_actors, agent, scheduler, train_iter, batch_size, min_exp_to_train,
        update_trigger=None, logger=None
    ):
        super().__init__(group_name, num_actors, agent, scheduler=scheduler, update_trigger=update_trigger)
        self.experience_pool = {
            agent: SimpleStore(["state", "action", "reward", "next_state", "loss"])
            for agent in agent.agent_dict
        }
        self.train_iter = train_iter   # Number of training iterations per round of training
        self.batch_size = batch_size   # Mini-batch size
        self.min_exp_to_train = min_exp_to_train   # Minimum number of experiences required for training
        self.logger = logger if logger else DummyLogger()

    def run(self):
        for exploration_params in self.scheduler:
            metrics_by_src, exp_by_src = self.collect(
                self.scheduler.iter, model_dict=self.agent.dump_model(), exploration_params=exploration_params
            )
            for src, metrics in metrics_by_src.items():
                self.logger.info(f"{src}.ep-{self.scheduler.iter}: {metrics} ({exploration_params})")
            # Store experiences for each agent
            for agent_id, exp in concat(exp_by_src).items():
                exp.update({"loss": [1e8] * len(list(exp.values())[0])})
                self.experience_pool[agent_id].put(exp)
                
            for agent_id, agent in self.agent.agent_dict.items():
                for i in range(self.train_iter):
                    if len(self.experience_pool[agent_id]) >= self.min_exp_to_train:
                        batch, indexes = self._get_batch(agent_id)
                        loss = agent.learn(*batch)
                        self.experience_pool[agent_id].update(indexes, {"loss": list(loss)})

            self.logger.info("Training finished")

    def _get_batch(self, agent_id):
        indexes, sample = self.experience_pool[agent_id].sample_by_key("loss", self.batch_size)
        states = np.asarray(sample["state"])
        actions = np.asarray(sample["action"])
        rewards = np.asarray(sample["reward"])
        next_states = np.asarray(sample["next_state"])
        return (states, actions, rewards, next_states), indexes
