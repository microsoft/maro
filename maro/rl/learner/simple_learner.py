# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from typing import Callable, Union

from maro.rl.actor.simple_actor import SimpleActor
from maro.rl.agent.simple_agent_manager import SimpleAgentManager
from maro.rl.dist_topologies.single_learner_multi_actor_sync_mode import ActorProxy
from maro.rl.explorer.abs_explorer import AbsExplorer
from maro.utils import DummyLogger, Logger
from maro.utils.exception.rl_toolkit_exception import InfiniteTrainingLoopError, InvalidEpisodeError

from .abs_learner import AbsLearner


class SimpleLearner(AbsLearner):
    """A simple implementation of ``AbsLearner``.

    Args:
        trainable_agents (AbsAgentManager): An AgentManager instance that manages all agents.
        actor (SimpleActor or ActorProxy): An SimpleActor or ActorProxy instance responsible for performing roll-outs
            (environment sampling).
        explorer (dict or AbsExplorer): An explorer instance responsible for generating exploration rates.
            Defaults to None.
        logger (Logger): Used to log important messages.
    """
    def __init__(
        self,
        trainable_agents: SimpleAgentManager,
        actor: Union[SimpleActor, ActorProxy],
        explorer: Union[dict, AbsExplorer] = None,
        logger: Logger = DummyLogger()
    ):
        super().__init__()
        self._trainable_agents = trainable_agents
        self._actor = actor
        self._explorer = explorer
        self._logger = logger
        self._performance_history = []

    def _get_epsilons(self, current_ep, max_ep):
        if self._explorer is None:
            return None
        elif isinstance(self._explorer, dict):
            return {
                agent_id: self._explorer[agent_id].generate_epsilon(current_ep, max_ep, self._performance_history)
                for agent_id in self._trainable_agents.agent_dict
            }
        else:
            return {
                agent_id: self._explorer.generate_epsilon(current_ep, max_ep, self._performance_history)
                for agent_id in self._trainable_agents.agent_dict
            }

    def _sample(self, ep, max_ep):
        """Perform one episode of environment sampling through actor roll-out."""
        model_dict = None if self._is_shared_agent_instance() else self._trainable_agents.dump_models()
        epsilon_dict = self._get_epsilons(ep, max_ep)
        performance, exp_by_agent = self._actor.roll_out(model_dict=model_dict, epsilon_dict=epsilon_dict)
        self._logger.info(f"ep {ep} - performance: {performance}, epsilons: {epsilon_dict}")
        return performance, exp_by_agent

    def train(
        self, max_episode: int, early_stopping_checker: Callable = None, warmup_ep: int = None,
        early_stopping_metric_func: Callable = None
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

        episode = 0
        metric_series = []
        while max_episode == -1 or episode < max_episode:
            performance, exp_by_agent = self._sample(episode, max_episode)
            latest = [perf for _, perf in performance] if isinstance(performance, list) else [performance]
            if early_stopping_checker is not None:
                metric_series.extend(map(early_stopping_metric_func, latest))
                if warmup_ep is None or episode >= warmup_ep and early_stopping_checker(metric_series):
                    self._logger.info("Early stopping condition hit. Training complete.")
                    break
            self._trainable_agents.train(exp_by_agent)
            episode += 1

    def test(self):
        """Test policy performance."""
        performance, _ = self._actor.roll_out(
            model_dict=self._trainable_agents.dump_models(),
            return_details=False
        )
        self._logger.info(f"test performance: {performance}")

    def exit(self, code: int = 0):
        """Tell the remote actor to exit"""
        if isinstance(self._actor, ActorProxy):
            self._actor.roll_out(done=True)
        sys.exit(code)

    def load_models(self, dir_path: str):
        self._trainable_agents.load_models_from_files(dir_path)

    def dump_models(self, dir_path: str):
        self._trainable_agents.dump_models_to_files(dir_path)

    def _is_shared_agent_instance(self):
        """If true, the set of agents performing inference in actor is the same as self._trainable_agents."""
        return isinstance(self._actor, SimpleActor) and id(self._actor.inference_agents) == id(self._trainable_agents)
