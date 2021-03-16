# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle

from numpy import asarray

from maro.rl import Learner, SimpleStore, concat
from maro.utils import DummyLogger


class BasicLearner(AbsLearner):
    def __init__(
        self, group_name, num_actors, agent, scheduler, train_iter, batch_size, min_exp_to_train,
        update_trigger=None, logger=None
    ):
        super().__init__(group_name, num_actors, agent, scheduler=scheduler, update_trigger=update_trigger)
        self.experience_pool = {agent: SimpleStore(["S", "A", "R", "S_", "loss"]) for agent in agent.agent_dict}
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
            self.store_experiences(exp_by_src)
            for i in range(self.train_iter):
                batch_by_agent, idx_by_agent = self.get_batch()
                loss_by_agent = {
                    agent_id: self.agent[agent_id].learn(*batch) for agent_id, batch in batch_by_agent.items()
                }
                self.update_loss_in_experience_pool(idx_by_agent, loss_by_agent)

            self.logger.info("Training finished")

        self._socket.send(b"DONE")

    def store_experiences(self, exp_by_src):
        """Store experiences in the experience pool."""
        for agent_id, exp in concat(exp_by_src).items():
            # ensure new experiences are sampled with the highest priority 
            exp.update({"loss": [1e8] * len(list(exp.values())[0])})
            self.experience_pool[agent_id].put(exp)

    def get_batch(self):
        idx, batch = {}, {}
        for agent_id, pool in self.experience_pool.items():
            if len(pool) < self.min_exp_to_train:
                continue
            indexes, sample = self.experience_pool[agent_id].sample_by_key("loss", self.batch_size)
            batch[agent_id] = asarray(sample["S"]), asarray(sample["A"]), asarray(sample["R"]), asarray(sample["S_"])
            idx[agent_id] = indexes

        return batch, idx

    def update_loss_in_experience_pool(self, idx_by_agent, loss_by_agent):
        for agent_id, loss in loss_by_agent.items():
            self.experience_pool[agent_id].update(idx_by_agent[agent_id], {"loss": list(loss)})
