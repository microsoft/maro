# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from collections import namedtuple
from typing import Union

from maro.rl.actor.simple_actor import SimpleActor
from maro.rl.agent.simple_agent_manager import SimpleAgentManager
from maro.rl.dist_topologies.single_learner_multi_actor_sync_mode import ActorProxy
from maro.rl.early_stopping.abs_early_stopping_checker import AbsEarlyStoppingChecker
from maro.rl.exploration.abs_exploration_scheduler import NullExplorationScheduler
from maro.utils import DummyLogger, Logger
from maro.utils.exception.rl_toolkit_exception import InfiniteTrainingLoopError, InvalidEpisodeError

from .abs_learner import AbsLearner

ExplorationOptions = namedtuple("ExplorationOptions", ["cls", "params"])


class SimpleLearner(AbsLearner):
    """A simple implementation of ``AbsLearner``.

    Args:
        agent_manager (AbsAgentManager): An AgentManager instance that manages all agents.
        actor (SimpleActor or ActorProxy): An SimpleActor or ActorProxy instance responsible for performing roll-outs
            (environment sampling).
        max_episode (int): number of episodes to be run. If -1, the training loop will run forever unless
            an ``early_stopping_checker`` is provided and the early stopping condition is met.
        exploration_options (ExplorationOptions): Exploration scheduler class and parameters. Defaults to None.
        early_stopping_checker (EarlyStoppingOptions): Early stopping checker that checks the performance history to
            determine whether early stopping condition is satisfied. Defaults to None.
        logger (Logger): Used to log important messages.
    """
    def __init__(
        self,
        agent_manager: SimpleAgentManager,
        actor: Union[SimpleActor, ActorProxy],
        max_episode: int,
        exploration_options: ExplorationOptions = None,
        early_stopping_checker: AbsEarlyStoppingChecker = None,
        logger: Logger = DummyLogger()
    ):
        if max_episode < -1:
            raise InvalidEpisodeError("max_episode can only be a non-negative integer or -1.")
        if max_episode == -1 and early_stopping_checker is None:
            raise InfiniteTrainingLoopError(
                "The training loop will run forever since neither maximum episode nor early stopping checker "
                "is provided. "
            )
        super().__init__()
        self._agent_manager = agent_manager
        self._actor = actor
        self._max_episode = max_episode
        if exploration_options is None:
            self._exploration_scheduler = NullExplorationScheduler(max_episode)
        else:
            self._exploration_scheduler = exploration_options.cls(max_ep=max_episode, **exploration_options.params)
        self._early_stopping_checker = early_stopping_checker
        self._logger = logger
        self._performance_history = []

    def learn(self):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        while True:
            try:
                exploration_params = next(self._exploration_scheduler)
                performance, exp_by_agent = self._actor.roll_out(
                    model_dict=None if self._is_shared_agent_instance() else self._agent_manager.dump_models(),
                    exploration_params=exploration_params
                )

                self._logger.info(
                    f"ep {self._exploration_scheduler.current_ep} - "
                    f"performance: {performance}, exploration_params: {exploration_params}"
                )

                # Early stopping checking
                if self._early_stopping_checker is not None and self._early_stopping_checker.update(performance):
                    self._logger.info("Early stopping condition hit. Training complete.")
                    break
            except StopIteration:
                self._logger.info(f"Maximum number of episodes ({self._max_episode}) reached. Training complete.")
                break

            self._agent_manager.train(exp_by_agent)

    def test(self):
        """Test policy performance."""
        performance, _ = self._actor.roll_out(
            model_dict=self._agent_manager.dump_models(),
            return_details=False
        )
        self._logger.info(f"test performance: {performance}")

    def exit(self, code: int = 0):
        """Tell the remote actor to exit."""
        if isinstance(self._actor, ActorProxy):
            self._actor.roll_out(done=True)
        sys.exit(code)

    def load_models(self, dir_path: str):
        self._agent_manager.load_models_from_files(dir_path)

    def dump_models(self, dir_path: str):
        self._agent_manager.dump_models_to_files(dir_path)

    def _is_shared_agent_instance(self):
        """If true, the set of agents performing inference in actor is the same as self._agent_manager."""
        return isinstance(self._actor, SimpleActor) and id(self._actor.agents) == id(self._agent_manager)
