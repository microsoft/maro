# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import AbsLearner
from maro.utils import LogFormat, Logger

from examples.cim.policy_optimization.components.actor import Actor


class Learner(AbsLearner):
    def __init__(self, actor, scheduler):
        super().__init__(actor, scheduler)
        self._logger = Logger("learner", format_=LogFormat.simple, auto_timestamp=False)

    def learn(self):
        for _ in self.scheduler:
            performance, exp = self.actor.roll_out(self.scheduler.iter)
            self._logger.info(f"ep {self.scheduler.iter} - performance: {performance}")
            self.scheduler.record_performance(performance)
            self.update(exp)

    def test(self):
        performance, _ = self.actor.roll_out(self.scheduler.iter, training=False)

    def update(self, experiences_by_agent):
        for agent_id, exp in experiences_by_agent.items():
            if not isinstance(exp, list):
                exp = [exp]
            for trajectory in exp:
                self.actor.agent[agent_id].train(
                    trajectory["state"],
                    trajectory["action"],
                    trajectory["log_action_prob"],
                    trajectory["reward"]
                )
