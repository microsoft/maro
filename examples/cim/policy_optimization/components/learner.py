# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import deque

from maro.rl import AbsLearner
from maro.utils import LogFormat, Logger

from examples.cim.policy_optimization.components.actor import Actor


class Learner(AbsLearner):
    """Learner with an early stopping mechanism.

    Args:
        actor An ``AbsActor`` instance that performs roll-outs.
        scheduler: A ``Scheduler`` instance that controls the training loop.
        warmup_ep: Episode from which early stopping checking is initiated.
        k: Number of consecutive fulfillment rates above the performance threshold required to trigger early stopping.
        perf_thresh: Performance threshold.
    """
    def __init__(self, actor, scheduler, warmup_ep=None, k=None, perf_thresh=None):
        super().__init__(actor, scheduler=scheduler)
        self._warmup_ep = warmup_ep
        self._k = k
        self._perf_thresh = perf_thresh
        self._perf_history = deque()
        self._logger = Logger("learner", format_=LogFormat.simple, auto_timestamp=False)

    def learn(self):
        for _ in self.scheduler:
            exp = self.actor.roll_out(self.scheduler.iter)
            metrics = self.actor.env.metrics
            self._logger.info(f"ep {self.scheduler.iter} - performance: {metrics}")
            fulfillment = 1 - metrics["container_shortage"] / metrics["order_requirements"]
            self._perf_history.append(fulfillment)
            if len(self._perf_history) > self._k:
                self._perf_history.popleft()
            if self.scheduler.iter >= self._warmup_ep and min(self._perf_history) >= self._perf_thresh:
                self._logger.info(
                    f"{self._k} consecutive fulfillment rates above threshold {self._perf_thresh}. Training complete"
                )
                break
            self.update(exp)

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
