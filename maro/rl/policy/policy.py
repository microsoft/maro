# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import namedtuple

from maro.rl.exploration import AbsExploration
from maro.rl.experience import ExperienceMemory, ExperienceSet


class TrainingLoopConfig:
    __slots__ = [
        "sampler_cls", "batch_size", "train_iters", "sampler_kwargs", "new_experience_trigger",
        "experience_memory_size_trigger"
    ]

    def __init__(
        self,
        sampler_cls,
        batch_size: int,
        train_iters: int,
        sampler_kwargs: dict = None,
        new_experience_trigger: int = 1,
        experience_memory_size_trigger: int = 1
    ):
        self.sampler_cls = sampler_cls
        self.batch_size = batch_size
        self.train_iters = train_iters
        self.sampler_kwargs = sampler_kwargs if sampler_kwargs else {}
        self.new_experience_trigger = new_experience_trigger
        self.experience_memory_size_trigger = experience_memory_size_trigger


class AbsFixedPolicy(ABC):
    """Abstract fixed policy class.

    Args:
        config: Settings for the algorithm.
    """
    def __init__(self):
        pass

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


class NullPolicy(AbsFixedPolicy):
    def choose_action(self, state):
        return None


class AbsCorePolicy(ABC):
    def __init__(self, experience_memory: ExperienceMemory, generic_config: TrainingLoopConfig, special_config):
        self.experience_memory = experience_memory
        self.generic_config = generic_config
        self.special_config = special_config
        sampler_cls, batch_size = generic_config.sampler_cls, generic_config.batch_size
        self.sampler = sampler_cls(experience_memory, batch_size, **generic_config.sampler_kwargs)
        self._last_experience_memory_size = 0
        self._ready_to_learn = False

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
        cur_experience_memory_size = len(self.experience_memory)
        print("cur exp mem size: ", cur_experience_memory_size)
        num_new = cur_experience_memory_size - self._last_experience_memory_size
        self._ready_to_learn = (
            num_new >= self.generic_config.new_experience_trigger and
            cur_experience_memory_size >= self.generic_config.experience_memory_size_trigger
        )

    def learn(self):
        if not self._ready_to_learn:
            return False

        self._last_experience_memory_size = len(self.experience_memory)
        exp_mem, sampler, config = self.experience_memory, self.sampler, self.generic_config
        for _ in range(self.generic_config.train_iters):
            self.step(self.experience_memory.get(self.sampler.sample()))
            return True

        return False

    @abstractmethod
    def step(self, experience_set: ExperienceSet):
        raise NotImplementedError

    def state(self):
        pass

    def update(self, policy_state):
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

    def learn(self):
        """Algorithm-specific training logic.

        The parameters are data to train the underlying model on. Algorithm-specific loss and optimization
        should be reflected here.
        """
        self.core_policy.learn()

    def update(self, policy_state):
        self.core_policy.load(policy_state)

    def load(self, path: str):
        self.core_policy.load(path)

    def save(self, path: str):
        self.core_policy.save(path)
