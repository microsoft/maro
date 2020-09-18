# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.actor.abstract_actor import RolloutMode
from .abstract_learner import AbstractLearner
from maro.rl.agent.agent_manager import AgentManager
from maro.utils import DummyLogger


class SimpleLearner(AbstractLearner):
    """
    A learner class that executes simple roll-out-and-train cycles.
    """

    def __init__(self, trainable_agents: AgentManager, actor, logger=DummyLogger(), seed: int = None):
        """
        seed (int): initial random seed value for the underlying simulator. If None, no manual seed setting   \n
                        is performed.
        Args:
            trainable_agents (dict or AgentManager): an AgentManager instance that manages all agents
            actor (Actor of ActorProxy): an Actor or VectorActorProxy instance.
            logger: used for logging important events
            seed (int): initial random seed value for the underlying simulator. If None, no seed fixing is done \n
                        for the underlying simulator.
        """
        assert isinstance(trainable_agents, AgentManager), \
            "SimpleLearner only accepts AgentManager for parameter trainable_agents"
        super().__init__(trainable_agents=trainable_agents, actor=actor, logger=logger)
        self._seed = seed

    def train(self, total_episodes):
        for current_ep in range(1, total_episodes+1):
            models = None if self._is_shared_agent_instance() else self._trainable_agents.get_models()
            performance, exp_by_agent = self._actor.roll_out(mode=RolloutMode.TRAIN,
                                                             models=models,
                                                             epsilon_dict=self._trainable_agents.explorer.epsilon,
                                                             seed=self._seed)
            if self._seed is not None:
                self._seed += len(performance)
            for actor_id, perf in performance.items():
                self._logger.info(f"ep {current_ep} - performance: {perf}, source: {actor_id}, "
                                  f"epsilons: {self._trainable_agents.explorer.epsilon}")

            self._trainable_agents.store_experiences(exp_by_agent)
            self._trainable_agents.train()
            self._trainable_agents.update_epsilon(performance)

    def test(self):
        performance, _ = self._actor.roll_out(mode=RolloutMode.TEST, models=self._trainable_agents.get_models())
        for actor_id, perf in performance.items():
            self._logger.info(f"test performance from {actor_id}: {perf}")
        self._actor.roll_out(mode=RolloutMode.EXIT)

    def dump_models(self, dir_path: str):
        self._trainable_agents.dump_models(dir_path)
