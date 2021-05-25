# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from os import getcwd
from typing import Union

from numpy import asarray

from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.scheduling import Scheduler
from maro.rl.storage import SimpleStore
from maro.rl.utils import ExperienceCollectionUtils
from maro.utils import Logger

from .actor import Actor
from .actor_proxy import ActorProxy


class AbsLearner(ABC):
    """Learner class.

    Args:
        actor (Union[Actor, ActorProxy]): ``Actor`` or ``ActorProxy`` instance responsible for collecting roll-out
            data for learning purposes. If it is an ``Actor``, it will perform roll-outs locally. If it is an
            ``ActorProxy``, it will coordinate a set of remote actors to perform roll-outs in parallel.
        agent (Union[AbsAgent, MultiAgentWrapper]): Learning agents. If None, the actor must be an ``Actor`` that
            contains actual agents, rather than an ``ActorProxy``. Defaults to None.
    """
    def __init__(
        self,
        actor: Union[Actor, ActorProxy],
        agent: Union[AbsAgent, MultiAgentWrapper] = None,
        log_dir: str = getcwd()
    ):
        super().__init__()
        if isinstance(actor, ActorProxy):
            assert agent, "agent cannot be None when the actor is a proxy."
            self.agent = agent
        else:
            # The agent passed to __init__ is ignored in this case
            self.agent = actor.agent
        self.actor = actor
        self.logger = Logger("LEARNER", dump_folder=log_dir)

    @abstractmethod
    def run(self):
        """Main learning loop is implemented here."""
        return NotImplementedError


class OnPolicyLearner(AbsLearner):
    def __init__(
        self,
        actor: Union[Actor, ActorProxy],
        max_episode: int,
        agent: Union[AbsAgent, MultiAgentWrapper] = None,
        log_dir: str = getcwd()
    ):
        super().__init__(actor, agent=agent, log_dir=log_dir)
        self.max_episode = max_episode

    def run(self):
        for ep in range(self.max_episode):
            env_metrics, exp = self.actor.roll_out(
                ep, model_by_agent=self.agent.dump_model() if isinstance(self.actor, ActorProxy) else None
            )
            self.logger.info(f"ep-{ep}: {env_metrics}")
            exp = ExperienceCollectionUtils.stack(
                exp,
                is_single_source=isinstance(self.actor, Actor),
                is_single_agent=isinstance(self.agent, AbsAgent)
            )
            if isinstance(self.agent, AbsAgent):
                for e in exp:
                    self.agent.learn(*e["args"], **e.get("kwargs", {}))
            else:
                for agent_id, ex in exp.items():
                    for e in ex:
                        self.agent[agent_id].learn(*e["args"], **e.get("kwargs", {}))

            self.logger.info("Agent learning finished")

        # Signal remote actors to quit
        if isinstance(self.actor, ActorProxy):
            self.actor.terminate()


MAX_LOSS = 1e8


class OffPolicyLearner(AbsLearner):
    def __init__(
        self,
        actor: Union[Actor, ActorProxy],
        scheduler: Scheduler,
        agent: Union[AbsAgent, MultiAgentWrapper] = None,
        train_iter: int = 1,
        min_experiences_to_train: int = 0,
        batch_size: int = 128,
        prioritized_sampling_by_loss: bool = False,
        log_dir: str = getcwd()
    ):
        super().__init__(actor, agent=agent, log_dir=log_dir)
        self.scheduler = scheduler
        if isinstance(self.agent, AbsAgent):
            self.experience_pool = SimpleStore(["S", "A", "R", "S_", "loss"])
        else:
            self.experience_pool = {
                agent: SimpleStore(["S", "A", "R", "S_", "loss"]) for agent in self.agent.agent_dict
            }
        self.train_iter = train_iter
        self.min_experiences_to_train = min_experiences_to_train
        self.batch_size = batch_size
        self.prioritized_sampling_by_loss = prioritized_sampling_by_loss

    def run(self):
        for exploration_params in self.scheduler:
            rollout_index = self.scheduler.iter
            env_metrics, exp = self.actor.roll_out(
                rollout_index,
                model_by_agent=self.agent.dump_model() if isinstance(self.actor, ActorProxy) else None,
                exploration_params=exploration_params
            )
            self.logger.info(f"ep-{rollout_index}: {env_metrics} ({exploration_params})")

            # store experiences in the experience pool.
            exp = ExperienceCollectionUtils.concat(
                exp,
                is_single_source=isinstance(self.actor, Actor),
                is_single_agent=isinstance(self.agent, AbsAgent)
            )
            if isinstance(self.agent, AbsAgent):
                exp.update({"loss": [MAX_LOSS] * len(list(exp.values())[0])})
                self.experience_pool.put(exp)
                for i in range(self.train_iter):
                    batch, idx = self.get_batch()
                    loss = self.agent.learn(*batch)
                    self.experience_pool.update(idx, {"loss": list(loss)})
            else:
                for agent_id, ex in exp.items():
                    # ensure new experiences are sampled with the highest priority
                    ex.update({"loss": [MAX_LOSS] * len(list(ex.values())[0])})
                    self.experience_pool[agent_id].put(ex)

                for i in range(self.train_iter):
                    batch_by_agent, idx_by_agent = self.get_batch()
                    loss_by_agent = {
                        agent_id: self.agent[agent_id].learn(*batch) for agent_id, batch in batch_by_agent.items()
                    }
                    for agent_id, loss in loss_by_agent.items():
                        self.experience_pool[agent_id].update(idx_by_agent[agent_id], {"loss": list(loss)})

            self.logger.info("Agent learning finished")

        # Signal remote actors to quit
        if isinstance(self.actor, ActorProxy):
            self.actor.terminate()

    def get_batch(self):
        if isinstance(self.agent, AbsAgent):
            if len(self.experience_pool) < self.min_experiences_to_train:
                return None, None
            if self.prioritized_sampling_by_loss:
                indexes, sample = self.experience_pool.sample_by_key("loss", self.batch_size)
            else:
                indexes, sample = self.experience_pool.sample(self.batch_size)
            batch = asarray(sample["S"]), asarray(sample["A"]), asarray(sample["R"]), asarray(sample["S_"])
            return batch, indexes
        else:
            idx, batch = {}, {}
            for agent_id, pool in self.experience_pool.items():
                if len(pool) < self.min_experiences_to_train:
                    continue
                if self.prioritized_sampling_by_loss:
                    indexes, sample = self.experience_pool[agent_id].sample_by_key("loss", self.batch_size)
                else:
                    indexes, sample = self.experience_pool[agent_id].sample(self.batch_size)
                batch[agent_id] = (
                    asarray(sample["S"]), asarray(sample["A"]), asarray(sample["R"]), asarray(sample["S_"])
                )
                idx[agent_id] = indexes

            return batch, idx
