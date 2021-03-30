# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from typing import Union

from maro.rl.actor.simple_actor import SimpleActor
from maro.rl.agent_manager import AbsAgentManager
from maro.rl.dist_topologies.single_learner_multi_actor_sync_mode import ActorProxy
from maro.rl.scheduling.scheduler import Scheduler
from maro.utils import DummyLogger, Logger

from maro.rl import AbsLearner


class VMLearner(AbsLearner):
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
        model_path: str,
        eval_interval: int,
        agent_manager: AbsAgentManager,
        actor: Union[SimpleActor, ActorProxy],
        scheduler: Scheduler,
        simulation_logger: Logger = DummyLogger(),
        test_simulation_logger: Logger = DummyLogger(),
        dqn_logger: Logger = DummyLogger(),
        test_dqn_logger: Logger = DummyLogger()
    ):
        super().__init__()
        self.model_path = model_path
        self.eval_interval = eval_interval
        self.agent_manager = agent_manager
        self._actor = actor
        self._scheduler = scheduler
        self._simulation_logger = simulation_logger
        self._test_simulation_logger = test_simulation_logger
        self._dqn_logger = dqn_logger
        self._test_dqn_logger = test_dqn_logger

    def learn(self):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        self.test_epoch(0)
        for exploration_params in self._scheduler:
            performance, exp_by_agent, reward = self._actor.roll_out(
                model_dict=None if self._is_shared_agent_instance() else self.agent_manager.dump_models(),
                exploration_params=exploration_params
            )
            self._scheduler.record_performance(performance)
            ep_summary = f"ep {self._scheduler.current_iter} - performance: {performance}"
            self._simulation_logger.info(ep_summary)

            loss, learning_rate = self.agent_manager.train(exp_by_agent)
            ep_summary = f"ep {self._scheduler.current_iter} - reward: {reward}, loss: {loss}, learning_rate: {learning_rate}, exploration_params: {exploration_params}"
            self._dqn_logger.info(ep_summary)

            if self._scheduler.current_iter % self.eval_interval == 0:
                self.test_epoch(self._scheduler.current_iter)

    def test_epoch(self, current_iter):
        """Test policy performance."""
        performance, _, reward = self._actor.roll_out(
            model_dict=self.agent_manager.dump_models(),
            return_details=False
        )

        ep_summary = f"ep {current_iter} - performance: {performance}"
        self._test_simulation_logger.info(ep_summary)

        ep_summary = f"ep {current_iter} - reward: {reward}"
        self._test_dqn_logger.info(ep_summary)

        self.dump_models(os.path.join(os.getcwd(), f"{self.model_path}/epoch_{current_iter}"))

    def test(self):
        """Test policy performance."""
        performance, _, reward = self._actor.roll_out(
            model_dict=self.agent_manager.dump_models(),
            return_details=False
        )
        
        ep_summary = f"ep test - performance: {performance}"
        self._test_simulation_logger.info(ep_summary)

        ep_summary = f"ep test - reward: {reward}"
        self._test_dqn_logger.info(ep_summary)

        self.dump_models(os.path.join(os.getcwd(), f"{self.model_path}/final"))

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
