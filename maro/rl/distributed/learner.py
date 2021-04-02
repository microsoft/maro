# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Callable, Union

from numpy import asarray

from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.scheduling import Scheduler
from maro.rl.storage import SimpleStore
from maro.utils import InternalLogger

from .actor_proxy import ActorProxy


class AbsDistLearner(ABC):
    """Learner class for distributed training.

    Args:
        actor_proxy (ActorProxy): ``ActorProxy`` instance that manages a set of remote actors to collect roll-out
            data for learning purposes.
        agent (Union[AbsAgent, MultiAgentWrapper]): Learning agents.
    """
    def __init__(
        self,
        actor_proxy: ActorProxy,
        agent: Union[AbsAgent, MultiAgentWrapper],
        scheduler: Scheduler,

    ):
        super().__init__()
        self.actor_proxy = actor_proxy
        self.agent = MultiAgentWrapper(agent) if isinstance(agent, AbsAgent) else agent
        self.logger = InternalLogger("LEARNER")

    def run(self):
        """Main learning loop is implemented here."""
        for exploration_params in self.scheduler:
            rollout_index = self.scheduler.iter
            env_metrics = self.actor_proxy.roll_out(
                rollout_index, model_by_agent=self.agent.dump_model(), exploration_params=exploration_params
            )
            self.logger.info(f"ep-{rollout_index}: {env_metrics} ({exploration_params})")

            for _ in range(self.train_iter):
                batch_by_agent, idx_by_agent = self.get_batch()
                for agent_id, batch in batch_by_agent.items():
                    self.agent[agent_id].learn(*batch)

            self.logger.info("Agent learning finished")

        # Signal remote actors to quit
        self.actor_proxy.terminate()


class OnPolicyDistLearner(AbsDistLearner):
    def __init__(self, actor_proxy: ActorProxy, agent: Union[AbsAgent, MultiAgentWrapper], max_episode: int):
        super().__init__(actor_proxy, agent)
        self.max_episode = max_episode

    def run(self):
        for ep in range(self.max_episode):
            env_metrics = self.actor_proxy.roll_out(ep, model_by_agent=self.agent.dump_model())
            self.logger.info(f"ep-{ep}: {env_metrics}")
            for agent_id, replay in self.actor_proxy.replay_memory.items():
                self.agent[agent_id].learn(replay["S"], replay["A"], replay["LOGP"], replay["R"], replay["S_"])

            self.logger.info("Agent learning finished")

        # Signal remote actors to quit
        self.actor_proxy.terminate()


class OffPolicyDistLearner(AbsDistLearner):
    def __init__(
        self,
        actor_proxy: ActorProxy,
        agent: Union[AbsAgent, MultiAgentWrapper],
        scheduler: Scheduler,
        train_iter: int = 1,
        min_experiences_to_train: int = 0,
        batch_size: int = 128
    ):
        super().__init__(actor_proxy, agent)
        self.scheduler = scheduler
        self.train_iter = train_iter
        self.min_experiences_to_train = min_experiences_to_train
        self.batch_size = batch_size

    def run(self):
        for exploration_params in self.scheduler:
            rollout_index = self.scheduler.iter
            env_metrics = self.actor_proxy.roll_out(
                rollout_index, model_by_agent=self.agent.dump_model(), exploration_params=exploration_params
            )
            self.logger.info(f"ep-{rollout_index}: {env_metrics} ({exploration_params})")

            for _ in range(self.train_iter):
                batch_by_agent, idx_by_agent = self.get_batch()
                for agent_id, batch in batch_by_agent.items():
                    self.agent[agent_id].learn(*batch)

            self.logger.info("Agent learning finished")

        # Signal remote actors to quit
        self.actor_proxy.terminate()
