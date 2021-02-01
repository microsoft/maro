# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from typing import Union

from maro.rl.actor.simple_actor import SimpleActor
from maro.rl.agent_manager import AbsAgentManager
from maro.rl.dist_topologies.single_learner_multi_actor_sync_mode import ActorProxy
from maro.rl.scheduling.scheduler import Scheduler
from maro.utils import DummyLogger, Logger

from .abs_learner import AbsLearner


class SimpleLearner(AbsLearner):
    """A simple implementation of ``AbsLearner``.

    Args:
        agent_manager (AbsAgentManager): An AgentManager instance that manages all agents.
        actor (SimpleActor or ActorProxy): An SimpleActor or ActorProxy instance responsible for performing roll-outs
            (environment sampling).
        scheduler (AbsScheduler): A scheduler responsible for iterating over episodes and generating exploration
            parameters if necessary.
        logger (Logger): Used to log important messages.
    """
    def __init__(
        self,
        agent_manager: AbsAgentManager,
        actor: Union[SimpleActor, ActorProxy],
        scheduler: Scheduler,
        logger: Logger = DummyLogger()
    ):
        super().__init__()
        self.agent_manager = agent_manager
        self._actor = actor
        self._scheduler = scheduler
        self._logger = logger

    def learn(self):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        for exploration_params in self._scheduler:
            performance, exp_by_agent = self._actor.roll_out(
                model_dict=None if self._is_shared_agent_instance() else self.agent_manager.dump_models(),
                exploration_params=exploration_params
            )
            self._scheduler.record_performance(performance)
            ep_summary = f"ep {self._scheduler.current_iter} - performance: {performance}"
            if exploration_params:
                ep_summary = f"{ep_summary}, exploration_params: {exploration_params}"
            self._logger.info(ep_summary)
            self.agent_manager.train(exp_by_agent)

    def test(self):
        """Test policy performance."""
        performance, _ = self._actor.roll_out(
            model_dict=self.agent_manager.dump_models(),
            return_details=False
        )
        self._scheduler.record_performance(performance)

    def exit(self, code: int = 0):
        """Tell the remote actor to exit."""
        if isinstance(self._actor, ActorProxy):
            self._actor.roll_out(done=True)
        sys.exit(code)

    def load_models(self, dir_path: str):
        self.agent_manager.load_models_from_files(dir_path)

    def dump_models(self, dir_path: str):
        self.agent_manager.dump_models_to_files(dir_path)

    def _is_shared_agent_instance(self):
        """If true, the set of agents performing inference in actor is the same as self.agent_manager."""
        return isinstance(self._actor, SimpleActor) and id(self._actor.agent_manager) == id(self.agent_manager)
