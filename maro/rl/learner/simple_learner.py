# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .abs_learner import AbsLearner
from maro.rl.agent.abs_agent_manager import AbsAgentManager
from maro.rl.actor.simple_actor import SimpleActor
from maro.utils import DummyLogger


class SimpleLearner(AbsLearner):
    """
    It is used to control the policy learning process...
    """
    def __init__(self, trainable_agents: AbsAgentManager, actor, logger=DummyLogger()):
        """
        seed (int): initial random seed value for the underlying simulator. If None, no manual seed setting is
                    performed.
        Args:
            trainable_agents (AbsAgentManager): an AgentManager instance that manages all agents.
            actor (Actor or ActorProxy): an Actor or VectorActorProxy instance.
            logger: used for logging important events.
        """
        super().__init__()
        self._trainable_agents = trainable_agents
        self._actor = actor
        self._logger = logger

    def train(self, total_episodes):
        """
        Main loop for collecting experiences and performance from the actor and using them to optimize models.
        Args:
            total_episodes (int): number of episodes for the main training loop.
        """
        for current_ep in range(1, total_episodes+1):
            model_dict = None if self._is_shared_agent_instance() else self._trainable_agents.get_models()
            epsilon_dict = self._trainable_agents.explorer.epsilon if self._trainable_agents.explorer else None
            performance, exp_by_agent = self._actor.roll_out(model_dict=model_dict, epsilon_dict=epsilon_dict)
            if isinstance(performance, dict):
                for actor_id, perf in performance.items():
                    self._logger.info(f"ep {current_ep} - performance: {perf}, source: {actor_id}, epsilons: {epsilon_dict}")
            else:
                self._logger.info(f"ep {current_ep} - performance: {performance}, epsilons: {epsilon_dict}")

            self._trainable_agents.store_experiences(exp_by_agent)
            self._trainable_agents.train()
            self._trainable_agents.update_epsilon(performance)

    def test(self):
        """
        This tells the actor to perform one episode of roll-out for model testing purposes.
        """
        performance, _ = self._actor.roll_out(model_dict=self._trainable_agents.get_models(), return_details=False)
        for actor_id, perf in performance.items():
            self._logger.info(f"test performance from {actor_id}: {perf}")
        self._actor.roll_out(done=True)

    def dump_models(self, dir_path: str):
        self._trainable_agents.dump_models(dir_path)

    def _is_shared_agent_instance(self):
        """
        If true, the set of agents performing inference in actor is the same as self._trainable_agents.
        """
        return isinstance(self._actor, SimpleActor) and id(self._actor.inference_agents) == id(self._trainable_agents)
