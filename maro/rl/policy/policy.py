# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import namedtuple

from maro.rl.exploration import AbsExploration
from maro.rl.experience import ExperienceMemory, ExperienceSet


class TrainingLoopConfig:
    __slots__ = [
        "sampler_cls", "batch_size", "train_iters", "sampler_kwargs", "new_experience_trigger",
        "num_warmup_experiences"
    ]

    def __init__(
        self,
        sampler_cls,
        batch_size: int,
        train_iters: int,
        sampler_kwargs: dict = None,
        new_experience_trigger: int = 1,
        num_warmup_experiences: int = 1
    ):
        self.sampler_cls = sampler_cls
        self.batch_size = batch_size
        self.train_iters = train_iters
        self.sampler_kwargs = sampler_kwargs if sampler_kwargs else {}
        self.new_experience_trigger = new_experience_trigger
        self.num_warmup_experiences = num_warmup_experiences


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
        self._num_new_exp = 0  # experience memory size when the last update was made
        self._warm_up = True
        self._update_ready = False

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
        self._num_new_exp += len(experience_set)
        self._warm_up = len(self.experience_memory) < self.generic_config.num_warmup_experiences
        self._update_ready = self._num_new_exp >= self.generic_config.new_experience_trigger

    def update(self):
        if self._warm_up or not self._update_ready:
            return False

        self._num_new_exp = 0
        for _ in range(self.generic_config.train_iters):
            self.learn(self.experience_memory.get(self.sampler.sample()))

        return True

    @abstractmethod
    def learn(self, experience_set: ExperienceSet):
        raise NotImplementedError

    def state(self):
        pass

    def load_state(self, policy_state):
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
