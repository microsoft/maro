# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from typing import Callable, Iterable, Union

from maro.rl.actor.simple_actor import SimpleActor
from maro.rl.agent.simple_agent_manager import SimpleAgentManager
from maro.rl.dist_topologies.single_learner_multi_actor_sync_mode import ActorProxy
from maro.utils import DummyLogger, Logger
from maro.utils.exception.rl_toolkit_exception import InfiniteTrainingLoopError, InvalidEpisodeError

from .abs_learner import AbsLearner


class SimpleLearner(AbsLearner):
    """A simple implementation of ``AbsLearner``.

    Args:
        agent_manager (AbsAgentManager): An AgentManager instance that manages all agents.
        actor (SimpleActor or ActorProxy): An SimpleActor or ActorProxy instance responsible for performing roll-outs
            (environment sampling).
        logger (Logger): Used to log important messages.
    """
    def __init__(
        self,
        agent_manager: SimpleAgentManager,
        actor: Union[SimpleActor, ActorProxy],
        logger: Logger = DummyLogger()
    ):
        super().__init__()
        self._agent_manager = agent_manager
        self._actor = actor
        self._logger = logger
        self._performance_history = []

    def learn(
        self,
        max_episode: int,
        early_stopping_checker: Callable = None,
        warmup_ep: int = None,
        early_stopping_metric_func: Callable = None,
    ):
        """Main loop for collecting experiences from the actor and using them to update policies.

        Args:
            max_episode (int): number of episodes to be run. If -1, the training loop will run forever unless
                an ``early_stopping_checker`` is provided and the early stopping condition is met.
            early_stopping_checker (Callable): A Callable object to determine whether the training loop should be
                terminated based on the latest performances. Defaults to None.
            warmup_ep (int): Episode from which early stopping check is initiated. Defaults to None.
            early_stopping_metric_func (Callable): A function to extract the metric from a performance record
                for early stopping checking. Defaults to None.
        """
        if max_episode < -1:
            raise InvalidEpisodeError("max_episode can only be a non-negative integer or -1.")
        if max_episode == -1 and early_stopping_checker is None:
            raise InfiniteTrainingLoopError(
                "The training loop will run forever since neither maximum episode nor early stopping checker "
                "is provided. "
            )
        if early_stopping_checker is not None:
            assert early_stopping_metric_func is not None, \
                "early_stopping_metric_func cannot be None if early_stopping_checker is provided."

        episode, metric_series = 0, []
        while max_episode == -1 or episode < max_episode:
            performance, exp_by_agent = self._actor.roll_out(
                model_dict=None if self._is_shared_agent_instance() else self._agent_manager.dump_models()
            )
            self._logger.info(f"ep {episode} - performance: {performance}")
            latest = [perf for _, perf in performance] if isinstance(performance, list) else [performance]
            if early_stopping_checker is not None:
                metric_series.extend(map(early_stopping_metric_func, latest))
                if warmup_ep is None or episode >= warmup_ep and early_stopping_checker(metric_series):
                    self._logger.info("Early stopping condition hit. Training complete.")
                    break
            self._agent_manager.train(exp_by_agent)
            episode += 1

    def learn_with_exploration_schedule(
        self,
        exploration_schedule: Union[Iterable, dict],
        early_stopping_checker: Callable = None,
        warmup_ep: int = None,
        early_stopping_metric_func: Callable = None,
    ):
        """Main loop for collecting experiences from the actor and using them to update policies.

        Args:
            exploration_schedule (Union[Iterable, dict]): Explorations schedules for the underlying agents. If it is
                a dictionary, the exploration schedule will be registered on a per-agent basis based on agent ID's.
                If it is a single iterable object, the exploration schedule will be registered for all agents.
            early_stopping_checker (Callable): A Callable object to determine whether the training loop should be
                terminated based on the latest performances. Defaults to None.
            warmup_ep (int): Episode from which early stopping check is initiated. Defaults to None.
            early_stopping_metric_func (Callable): A function to extract the metric from a performance record
                for early stopping checking. Defaults to None.
        """
        if early_stopping_checker is not None:
            assert early_stopping_metric_func is not None, \
                "early_stopping_metric_func cannot be None if early_stopping_checker is provided."

        self._agent_manager.register_exploration_schedule(exploration_schedule)
        ep, metric_series = 0, []
        while True:
            try:
                self._agent_manager.update_exploration_params()
                exploration_params = self._agent_manager.dump_exploration_params()
                if self._is_shared_agent_instance():
                    performance, exp_by_agent = self._actor.roll_out()
                else:
                    performance, exp_by_agent = self._actor.roll_out(
                        model_dict=self._agent_manager.dump_models(), exploration_params=exploration_params
                    )
                self._logger.info(f"performance: {performance}, exploration_params: {exploration_params}")
                # Early stopping checking
                latest = [perf for _, perf in performance] if isinstance(performance, list) else [performance]
                if early_stopping_checker is not None:
                    metric_series.extend(map(early_stopping_metric_func, latest))
                    if warmup_ep is None or ep >= warmup_ep and early_stopping_checker(metric_series):
                        self._logger.info("Early stopping condition hit. Training complete.")
                        break
            except StopIteration:
                self._logger.info(f"Maximum number of episodes {ep + 1} reached. Training complete.")
                break

            ep += 1
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
