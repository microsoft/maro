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
        learning_interval: Union[int, Dict[str, int]] = -1,
        end_of_episode_learning: bool = True   
    ):
        super().__init__()
        self.env = env
        self.agent = MultiAgentWrapper(agent) if isinstance(agent, AbsAgent) else agent
        self.scheduler = scheduler
        self.online = learning_interval != -1
        if isinstance(learning_interval, int):
            assert learning_interval == -1 or learning_interval > 0, \
                f"learning_interval must be -1 or a positive integer"
            self.learning_interval = {agent_id: learning_interval for agent_id in self.agent.names} 
        else:
            self.learning_interval = learning_interval
        self.end_of_episode_learning = end_of_episode_learning
        self.logger = InternalLogger("LEARNER")

    def run(self):
        for exploration_params in self.scheduler:
            # t0 = time.time()
            self.env.reset()
            if exploration_params:
                self.agent.set_exploration_params(exploration_params)

            pending_learning_agents = defaultdict(list)
            for agent_id, interval in self.learning_interval.items():
                pending_learning_agents[interval].append(agent_id)

            state = self.env.start(rollout_index=self.scheduler.iter)  # get initial state
            while state:
                action = self.agent.choose_action(state)
                state = self.env.step(action)
                if self.online and self.env.step_index in pending_learning_agents:
                    self.agent.store_experiences(self.env.get_experiences())
                    for agent_id in pending_learning_agents[self.env.step_index]:
                        self.agent.learn(agent_id)
                        next_learning_time = self.env.step_index + self.learning_interval[agent_id]
                        if next_learning_time > self.env.step_index:
                            pending_learning_agents[next_learning_time].append(agent_id)
                    del pending_learning_agents[self.env.step_index]
                    print(f"step = {self.env.step_index}, next up: {pending_learning_agents}")

            if self.end_of_episode_learning:
                self.agent.store_experiences(self.env.get_experiences())
                self.agent.learn()

            self.logger.info(f"ep-{self.scheduler.iter}: {self.env.metrics} ({exploration_params})")

            # t1 = time.time()
            # print(f"roll-out time: {t1 - t0}")
