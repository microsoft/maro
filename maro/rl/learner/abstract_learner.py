# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Union

from maro.rl.actor.simple_actor import SimpleActor
from maro.rl.dist_topologies.multi_actor_single_learner_sync import ActorProxy
from maro.rl.agent.agent_manager import AgentManager
from maro.utils import DummyLogger


class AbstractLearner(ABC):
    def __init__(self, *,
                 trainable_agents: Union[dict, AgentManager],
                 actor: Union[SimpleActor, ActorProxy],
                 logger=DummyLogger()
                 ):
        """
        This class contains the main driver logic for a RL task.
        Args:
            trainable_agents (dict or AgentManager): a dict of individual agents or an AgentManager instance that
                                                     manages all agents
            actor (Actor of ActorProxy): an Actor or VectorActorProxy instance.
            logger: used for logging important messages
        """
        self._trainable_agents = trainable_agents
        self._actor = actor
        self._logger = logger

    @abstractmethod
    def train(self, total_episodes):
        """
        Main loop for collecting experiences and performance from the actor and using them to optimize models
        Args:
            total_episodes (int): number of episodes for the main training loop
        """
        return NotImplementedError

    @abstractmethod
    def test(self):
        """
        Tells the actor to perform one episode of roll-out for model testing purposes
        """
        return NotImplementedError

    def train_test(self, total_episodes):
        self.train(total_episodes)
        self.test()

    def _is_shared_agent_instance(self):
        """
        returns True if the the set of agents performing inference in actor is the same as self._trainable_agents
        """
        return isinstance(self._actor, SimpleActor) and id(self._actor.inference_agents) == id(self._trainable_agents)

    @abstractmethod
    def dump_models(self, dir_path: str):
        return NotImplemented

    # TODO: add load models
