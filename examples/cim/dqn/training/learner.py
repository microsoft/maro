# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl import AbsLearner, concat
from maro.utils import DummyLogger


class BasicLearner(AbsLearner):
    def __init__(self, group_name, num_actors, agent, scheduler, update_trigger=None, logger=None):
        super().__init__(
            group_name, num_actors, agent,
            scheduler=scheduler,
            update_trigger=update_trigger
        )
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
                self.agent[agent_id].store_experiences(exp)

            for agent in self.agent.agent_dict.values():
                agent.train()

            self.logger.info("Training finished")
