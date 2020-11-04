# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.actor.simple_actor import SimpleActor
from maro.rl.agent.abs_agent_manager import AbsAgentManager
from maro.utils import DummyLogger

from .abs_learner import AbsLearner


class SimpleLearner(AbsLearner):
    """A simple implementation of ``AbsLearner``.

    Args:
        trainable_agents (AbsAgentManager): an AgentManager instance that manages all agents.
        actor (Actor or ActorProxy): an Actor or VectorActorProxy instance.
        logger: used for logging important messages.
    """
    def __init__(self, trainable_agents: AbsAgentManager, actor, logger=DummyLogger()):
        super().__init__()
        self._trainable_agents = trainable_agents
        self._actor = actor
        self._logger = logger

    def train(self, total_episodes):
        """Main loop for collecting experiences from the actor and using them to update policies.

        Args:
            total_episodes (int): number of episodes to be run.
        """
        for current_ep in range(1, total_episodes + 1):
            model_dict = None if self._is_shared_agent_instance() else self._trainable_agents.get_models()
            epsilon_dict = self._trainable_agents.explorer.epsilon if self._trainable_agents.explorer else None
            performance, exp_by_agent = self._actor.roll_out(model_dict=model_dict, epsilon_dict=epsilon_dict)
            self._logger.info(f"ep {current_ep} - performance: {performance}, epsilons: {epsilon_dict}")

            self._trainable_agents.store_experiences(exp_by_agent)
            self._trainable_agents.train()
            self._trainable_agents.update_epsilon(performance)

    def test(self):
        """Test policy performance."""
        performance, _ = self._actor.roll_out(model_dict=self._trainable_agents.get_models(), return_details=False)
        for actor_id, perf in performance.items():
            self._logger.info(f"test performance from {actor_id}: {perf}")
        self._actor.roll_out(done=True)

    def dump_models(self, dir_path: str):
        """Dump agents' models to disk."""
        self._trainable_agents.dump_models(dir_path)

    def _is_shared_agent_instance(self):
        """If true, the set of agents performing inference in actor is the same as self._trainable_agents."""
        return isinstance(self._actor, SimpleActor) and id(self._actor.inference_agents) == id(self._trainable_agents)
