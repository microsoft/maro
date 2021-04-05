# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from collections import defaultdict
from typing import Dict, Union

from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.scheduling import Scheduler
from maro.utils import InternalLogger

from .env_wrapper import AbsEnvWrapper


class Learner(object):
    """Learner class for distributed training.

    Args:
        env (AbsEnvWrapper): An ``AbsEnvWrapper`` instance that wraps an ``Env`` instance with scenario-specific
            processing logic and stores transitions during roll-outs in a replay memory.
        agent (Union[AbsAgent, MultiAgentWrapper]): Agent that interacts with the environment.
    """
    def __init__(
        self,
        env: AbsEnvWrapper,
        agent: Union[AbsAgent, MultiAgentWrapper],
        scheduler: Scheduler,
        agent_update_interval: int = -1
    ):
        super().__init__()
        if agent_update_interval == 0:
            raise ValueError("agent_update_interval must be a positive integer or None.")
        self.env = env
        self.agent = MultiAgentWrapper(agent) if isinstance(agent, AbsAgent) else agent
        self.scheduler = scheduler
        self.agent_update_interval = agent_update_interval
        self.logger = InternalLogger("LEARNER")

    def run(self):
        for exploration_params in self.scheduler:
            # t0 = time.time()
            self.env.reset()
            if exploration_params:
                self.agent.set_exploration_params(exploration_params)

            self.env.start(rollout_index=self.scheduler.iter)  # get initial state
            while self.env.state:
                action = self.agent.choose_action(self.env.state)
                self.env.step(action)
                if self.agent_update_interval != -1 and self.env.step_index % self.agent_update_interval == 0:
                    self.agent.update(self.env.pull_experiences())

            self.agent.update(self.env.pull_experiences())
            self.logger.info(f"ep-{self.scheduler.iter}: {self.env.metrics} ({exploration_params})")

            # t1 = time.time()
            # print(f"roll-out time: {t1 - t0}")
