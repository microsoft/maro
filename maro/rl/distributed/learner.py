# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from collections import defaultdict
from os import getcwd
from typing import Union

from maro.communication import Message, Proxy, SessionType
from maro.rl.agent import AbsAgent, AgentManager
from maro.rl.scheduling import Scheduler
from maro.utils import Logger

from .actor_manager import ActorManager
from .message_enums import MsgTag, MsgKey


class DistLearner(object):
    """Learner class for distributed training.

    Args:
        agent (Union[AbsAgent, AgentManager]): Learning agents.
        scheduler (Scheduler): A ``Scheduler`` instance for generating exploration parameters.
    """
    def __init__(
        self,
        agent: Union[AbsAgent, AgentManager],
        scheduler: Scheduler,
        actor_manager: ActorManager,
        agent_update_interval: int = -1,
        required_actor_finishes: str = None,
        discard_stale_experiences: bool = True,
        log_dir: str = getcwd()
    ):
        super().__init__()
        self.agent = AgentManager(agent) if isinstance(agent, AbsAgent) else agent
        self.scheduler = scheduler
        self.actor_manager = actor_manager
        self.agent_update_interval = agent_update_interval
        self.required_actor_finishes = required_actor_finishes
        self.discard_stale_experiences = discard_stale_experiences
        self._total_learning_time = 0
        self._logger = Logger("LEARNER", dump_folder=log_dir)

    def run(self):
        """Main learning loop."""
        t0 = time.time()
        for exploration_params in self.scheduler:
            updated_agents, num_actor_finishes, segment_index = self.agent.names, 0, 0
            while num_actor_finishes < self.required_actor_finishes:
                for exp, done in self.actor_manager.collect(
                    self.scheduler.iter,
                    segment_index,
                    self.agent_update_interval,
                    models=self.agent.dump_model(agent_ids=updated_agents),
                    exploration_params=exploration_params if segment_index == 0 else None,
                    required_actor_finishes=self.required_actor_finishes,
                    discard_stale_experiences=self.discard_stale_experiences
                ):
                    tl0 = time.time()
                    updated_agents = self.agent.learn(exp)
                    num_actor_finishes += done
                    self._total_learning_time += time.time() - tl0
                    self._logger.debug(f"total running time: {time.time() - t0}")
                    self._logger.debug(f"total learning time: {self._total_learning_time}")
                    self._logger.debug(f"total env steps: {self.actor_manager.total_env_steps}")
                    self._logger.info(f"total experiences collected: {self.actor_manager.total_experiences_collected}")

                segment_index += 1

        self.actor_manager.exit()
