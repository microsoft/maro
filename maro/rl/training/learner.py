# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from collections import defaultdict
from typing import Dict, Union

from maro.rl.agent import AbsPolicy, MultiAgentWrapper
from maro.rl.scheduling import Scheduler
from maro.utils import InternalLogger

from .env_wrapper import AbsEnvWrapper


class Learner(object):
    """Learner class for distributed training.

    Args:
        env (AbsEnvWrapper): An ``AbsEnvWrapper`` instance that wraps an ``Env`` instance with scenario-specific
            processing logic and stores transitions during roll-outs in a replay memory.
        agent (Union[AbsPolicy, MultiAgentWrapper]): Agent that interacts with the environment.
    """
    def __init__(
        self,
        env: AbsEnvWrapper,
        agent: Union[AbsPolicy, MultiAgentWrapper],
        scheduler: Scheduler,
        agent_update_interval: int = -1,
        log_env_metrics: bool = False
    ):
        super().__init__()
        if agent_update_interval == 0:
            raise ValueError("agent_update_interval must be a positive integer or None.")
        self.env = env
        self.agent = MultiAgentWrapper(agent) if isinstance(agent, AbsPolicy) else agent
        self.scheduler = scheduler
        self.agent_update_interval = agent_update_interval
        self.total_env_steps = 0
        self.total_experiences_collected = 0
        self.total_learning_time = 0
        self._log_env_metrics = log_env_metrics
        self._logger = InternalLogger("LEARNER")

    def run(self):
        t0 = time.time()
        for exploration_params in self.scheduler:
            self.env.reset()
            if exploration_params:
                self.agent.set_exploration_params(exploration_params)

            self.env.start(rollout_index=self.scheduler.iter)  # get initial state
            while self.env.state:
                action = self.agent.choose_action(self.env.state)
                self.env.step(action)
                if (
                    not self.env.state or
                    self.agent_update_interval != -1 and self.env.step_index % self.agent_update_interval == 0
                ):
                    exp, num_exp = self.env.pull_experiences()
                    tl0 = time.time()
                    self.agent.learn(exp)
                    self.total_learning_time += time.time() - tl0
                    self.total_env_steps += self.agent_update_interval
                    self.total_experiences_collected += num_exp
                    self._logger.debug(f"total running time: {time.time() - t0}")
                    self._logger.debug(f"total learning time: {self.total_learning_time}")
                    self._logger.debug(f"total env steps: {self.total_env_steps}")
                    self._logger.info(f"total experiences collected: {self.total_experiences_collected}")
                    if not self.env.state:
                        self._logger.info(f"total reward: {self.env.total_reward}")

            if self._log_env_metrics:
                self._logger.info(f"ep-{self.scheduler.iter}: {self.env.metrics} ({exploration_params})")
