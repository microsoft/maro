# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Callable, Union

from numpy import asarray

from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.scheduling import Scheduler
from maro.rl.storage import SimpleStore
from maro.utils import InternalLogger

from .env_wrapper import AbsEnvWrapper


class AbsLearner(ABC):
    """Learner class for distributed training.

    Args:
        env (AbsEnvWrapper): An ``AbsEnvWrapper`` instance that wraps an ``Env`` instance with scenario-specific
            processing logic and stores transitions during roll-outs in a replay memory.
        agent (Union[AbsAgent, MultiAgentWrapper]): Agent that interacts with the environment.
    """
    def __init__(self, env: AbsEnvWrapper, agent: Union[AbsAgent, MultiAgentWrapper]):
        super().__init__()
        self.env = env
        self.agent = MultiAgentWrapper(agent) if isinstance(agent, AbsAgent) else agent
        self.logger = InternalLogger("LEARNER")

    def roll_out(self, index: int, training: bool = True):
        self.env.reset()
        if not training:
            self.env.save_replay = False  # no need to record the trajectory if roll-out is not for training

        state = self.env.start(rollout_index=index)  # get initial state
        while state:
            action = self.agent.choose_action(state)
            state = self.env.step(action)

    @abstractmethod
    def run(self):
        """Main learning loop is implemented here."""
        return NotImplementedError


class OnPolicyLearner(AbsLearner):
    def __init__(self, env: AbsEnvWrapper, agent: Union[AbsAgent, MultiAgentWrapper], max_episode: int):
        super().__init__(env, agent)
        self.max_episode = max_episode

    def run(self):
        for ep in range(self.max_episode):
            self.roll_out(ep)
            self.logger.info(f"ep-{ep}: {self.env.metrics}")
            for agent_id, replay in self.env.replay_memory.items():
                self.agent[agent_id].learn(
                    asarray(replay["S"]),
                    asarray(replay["A"]),
                    asarray(replay["LOGP"]),
                    asarray(replay["R"]),
                    asarray(replay["S_"])
                )

            self.logger.info("Agent learning finished")


class OffPolicyLearner(AbsLearner):
    def __init__(
        self,
        env: AbsEnvWrapper,
        agent: Union[AbsAgent, MultiAgentWrapper],
        scheduler: Scheduler,
        train_iter: int = 1,
        min_experiences_to_train: int = 0,
        batch_size: int = 128
    ):
        super().__init__(env, agent)
        self.scheduler = scheduler
        self.train_iter = train_iter
        self.min_experiences_to_train = min_experiences_to_train
        self.batch_size = batch_size
        self.replay_memory = defaultdict(
            lambda: SimpleStore(capacity=replay_memory_size, overwrite_type=replay_memory_overwrite_type)
        )

    def run(self):
        for exploration_params in self.scheduler:
            rollout_index = self.scheduler.iter
            self.roll_out(rollout_index)
            self.logger.info(f"ep-{rollout_index}: {self.env.metrics} ({exploration_params})")
            # Add the latest transitions to the replay memory
            for agent_id, mem in self.env.replay_memory.items():
                self.replay_memory[agent_id].put(mem)

            # Training
            for _ in range(self.train_iter):
                batch_by_agent, idx_by_agent = self.get_batch()
                for agent_id, batch in batch_by_agent.items():
                    self.agent[agent_id].learn(*batch)

            self.logger.info("Agent learning finished")

    def get_batch(self):
        idx, batch = {}, {}
        for agent_id, mem in self.replay_memory.items():
            if len(mem) < self.min_experiences_to_train:
                continue
            indexes, sample = mem.sample(self.batch_size)
            batch[agent_id] = (
                asarray(sample["S"]), asarray(sample["A"]), asarray(sample["R"]), asarray(sample["S_"])
            )
            idx[agent_id] = indexes

        return batch, idx
