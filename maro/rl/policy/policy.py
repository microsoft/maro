# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import namedtuple

from maro.rl.exploration import AbsExploration
from maro.rl.experience import ExperienceMemory, ExperienceSet


class TrainingLoopConfig:
    __slots__ = ["sampler_cls", "batch_size", "train_iters", "sampler_kwargs"]

    def __init__(
        self,
        sampler_cls,
        batch_size: int,
        train_iters: int,
        sampler_kwargs: dict = None
    ):
        self.sampler_cls = sampler_cls
        self.batch_size = batch_size
        self.train_iters = train_iters
        self.sampler_kwargs = sampler_kwargs if sampler_kwargs else {}


class AbsPolicy(ABC):
    """Abstract fixed policy class.

    Args:
        config: Settings for the algorithm.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def choose_action(self, state):
        """Compute an action based on a state object.

        Args:
            state: State object.

        Returns:
            The action to be taken given the state object. It is usually necessary to use an ``ActionShaper`` to convert
            this to an environment executable action.
        """
        raise NotImplementedError


class NullPolicy(AbsPolicy):
    def choose_action(self, state):
        return None


class AbsCorePolicy(AbsPolicy):
    def __init__(
        self,
        experience_memory: ExperienceMemory,
        generic_config: TrainingLoopConfig,
        special_config
    ):
        super().__init__()
        self.experience_memory = experience_memory
        self.generic_config = generic_config
        self.special_config = special_config
        sampler_cls, batch_size = generic_config.sampler_cls, generic_config.batch_size
        self.sampler = sampler_cls(experience_memory, batch_size, **generic_config.sampler_kwargs)

    @abstractmethod
    def choose_action(self, state):
        """Compute an action based on a state object.

        Args:
            state: State object.

        Returns:
            The action to be taken given the state object. It is usually necessary to use an ``ActionShaper`` to convert
            this to an environment executable action.
        """
        raise NotImplementedError

    def store_experiences(self, experience_set: ExperienceSet):
        self.experience_memory.put(experience_set)

    def update(self):
        for _ in range(self.generic_config.train_iters):
            indexes, sp = self.sampler.sample()
            step_info = self.step(sp)
            self.sampler.update(indexes, step_info)

    @abstractmethod
    def step(self, experience_set: ExperienceSet):
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, policy_state):
        pass

    def load(self, path: str):
        pass

    def save(self, path: str):
        pass


class RLPolicy(object):
    """Abstract fixed policy class.

    Args:
        config: Settings for the algorithm.
    """
    def __init__(self, core_policy: AbsCorePolicy, exploration: AbsExploration = None):
        self.core_policy = core_policy
        self.exploration = exploration

    def choose_action(self, state):
        """Compute an action based on a state object.

        Args:
            state: State object.

        Returns:
            The action to be taken given the state object. It is usually necessary to use an ``ActionShaper`` to convert
            this to an environment executable action.
        """
        action = self.core_policy.choose_action(state)
        return self.exploration(action)        

    def update(self):
        """Algorithm-specific training logic.

        The parameters are data to train the underlying model on. Algorithm-specific loss and optimization
        should be reflected here.
        """
        self.core_policy.update()

    def learn(self, experience_set: ExperienceSet):
        return self.core_policy.learn(experience_set)

    def load_state(self, policy_state):
        self.core_policy.load_state(policy_state)

    def load(self, path: str):
        self.core_policy.load(path)

    def save(self, path: str):
        self.core_policy.save(path)
