# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import AbsDistLearner, concat_by_agent
from maro.utils import LogFormat, Logger


class SimpleDistLearner(AbsDistLearner):
    def learn(self):
        for exploration_params in self.scheduler:
            self.agent.set_exploration_params(exploration_params)
            performance, experiences = self.collect(exploration_params=exploration_params)
            for src, perf in performance.items():
                self._logger.info(
                    f"{src}.ep-{self.scheduler.iter} - performance: {perf}, exploration_params: {exploration_params}"
                )
            self.update(experiences)

    def update(self, exp_dict):
        # Store experiences for each agent
        for agent_id, exp in concat_by_agent(exp_dict).items():
            exp.update({"loss": [1e8] * len(list(exp.values())[0])})
            self.agent[agent_id].store_experiences(exp)

        for agent in self.agent.agent_dict.values():
            agent.train()

        self._logger.info("Training finished")
