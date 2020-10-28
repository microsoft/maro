# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable
import warnings

from .abs_learner import AbsLearner
from maro.rl.actor.simple_actor import SimpleActor
from maro.rl.agent.simple_agent_manager import SimpleAgentManager
from maro.rl.explorer.abs_explorer import AbsExplorer
from maro.utils import DummyLogger


class SimpleLearner(AbsLearner):
    """A simple implementation of ``AbsLearner``.

    Args:
        trainable_agents (AbsAgentManager): an AgentManager instance that manages all agents.
        actor (Actor or ActorProxy): an Actor or VectorActorProxy instance.
        logger: used for logging important messages.
    """
    def __init__(
        self,
        trainable_agents: SimpleAgentManager,
        actor,
        explorer: AbsExplorer = None,
        logger=DummyLogger()
    ):
        super().__init__()
        self._trainable_agents = trainable_agents
        self._actor = actor
        self._explorer = explorer
        self._logger = logger
        self._performance_history = []

    def _get_epsilons(self, current_ep, max_ep):
        if self._explorer is not None:
            return {
                agent_id: self._explorer.generate_epsilon(current_ep, max_ep, self._performance_history)
                for agent_id in self._trainable_agents.agent_dict
            }
        else:
            return None

    def _sample(self, ep, max_ep):
        """One episode"""
        model_dict = None if self._is_shared_agent_instance() else self._trainable_agents.dump_models()
        epsilon_dict = self._get_epsilons(ep, max_ep)
        performance, exp_by_agent = self._actor.roll_out(model_dict=model_dict, epsilon_dict=epsilon_dict)
        self._logger.info(f"ep {ep} - performance: {performance}, epsilons: {epsilon_dict}")
        return performance, exp_by_agent

    def train(self, max_episode: int, early_stopping_checker: Callable = None):
        """Main loop for collecting experiences from the actor and using them to update policies.

        Args:
            max_episode (int): number of episodes to be run. If negative, the training loop will run forever unless
                an ``early_stopping_checker`` is provided and the early stopping condition is met.
            early_stopping_checker (Callable): A Callable object to judge whether the training loop should be ended
                based on the latest performances.
        """
        if max_episode < 0:
            if early_stopping_checker is None:
                warnings.warn("No max episode and early stopping checker provided. The training loop will run forever.")
            episode = 1
            while True:
                performance, exp_by_agent = self._sample(episode, max_episode)
                self._performance_history.append(performance)
                if early_stopping_checker is not None and early_stopping_checker(self._performance_history):
                    break
                episode += 1
        else:
            for episode in range(1, max_episode + 1):
                performance, exp_by_agent = self._sample(episode, max_episode)
                self._performance_history.append(performance)
                if early_stopping_checker is not None and early_stopping_checker(self._performance_history):
                    break
                self._trainable_agents.train(exp_by_agent)

    def test(self):
        """Test policy performance."""
        performance, _ = self._actor.roll_out(
            model_dict=self._trainable_agents.dump_models(),
            return_details=False
        )
        self._logger.info(f"test performance: {performance}")
        self._actor.roll_out(done=True)

    def dump_models(self, model_dump_dir: str):
        self._trainable_agents.dump_models_to_files(model_dump_dir)

    def _is_shared_agent_instance(self):
        """If true, the set of agents performing inference in actor is the same as self._trainable_agents."""
        return isinstance(self._actor, SimpleActor) and id(self._actor.inference_agents) == id(self._trainable_agents)
