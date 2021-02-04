# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import AbsLearner
from maro.utils import LogFormat, Logger


class Learner(AbsLearner):
    def __init__(self, actor, scheduler):
        super().__init__(actor, scheduler)
        self._logger = Logger("learner", format_=LogFormat.simple, auto_timestamp=False)

    def learn(self):
        for exploration_params in self.scheduler:
            # load exploration parameters
            self.actor.agent.set_exploration_params(exploration_params)
            performance, exp = self.actor.roll_out(self.scheduler.iter)
            self._logger.info(
                f"ep {self.scheduler.iter} - performance: {performance}, exploration_params: {exploration_params}"
            )
            self.update(exp)

    def test(self):
        performance, _ = self.actor.roll_out(self.scheduler.iter, is_training=False)

    def update(self, experiences_by_agent):
        # Store experiences for each agent
        for agent_id, exp in experiences_by_agent.items():
            exp.update({"loss": [1e8] * len(list(exp.values())[0])})
            self.actor.agent[agent_id].store_experiences(exp)

        for agent in self.actor.agent.agent_dict.values():
            agent.train()
