# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import DistLearner, InferenceLearner, concat_experiences_by_agent
from maro.utils import LogFormat, Logger


class SimpleDistLearner(DistLearner):
    def learn(self):
        for exploration_params in self._scheduler:
            self.request_rollout(exploration_params=exploration_params)
            performance, experiences = self.collect()
            for src, perf in performance.items():
                self._logger.info(
                    f"{src}.ep-{self.scheduler.iter} - performance: {perf}, exploration_params: {exploration_params}"
                )
    
    def update(self, exp_dict):
        exp_dict = concat_experiences_by_agent(exp_dict)
        for agent_id, exp in exp_dict.items():
            self.agent[agent_id].train(exp)
        self._logger.info("Training finished")
