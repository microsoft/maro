# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

from maro.rl.algorithms import AbsAlgorithm
from maro.rl.experience import ExperienceMemory, ExperienceSet, UniformSampler


class AbsPolicy(ABC):
    """Abstract policy class."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError


class NullPolicy(AbsPolicy):
    """Dummy policy that does nothing.

    Note that the meaning of a "None" action may depend on the scenario.
    """
    def choose_action(self, state):
        return None


class CorePolicy(AbsPolicy):
    """Policy that can update itself using simulation experiences.

    Reinforcement learning (RL) policies should inherit from this.

    Args:
        algorithm (AbsAlgorithm): Algorithm instance.
        memory_capacity (int): Capacity for the internal experience memory.
        random_overwrite (bool): This specifies overwrite behavior when the capacity is reached. If this is True,
            overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
            wrap-around. Defaults to False.
        sampler_cls: Type of experience sampler. Must be a subclass of ``AbsSampler``. Defaults to ``UnifromSampler``.
        sampler_kwargs (dict): Keyword arguments for ``sampler_cls``.
    """
    def __init__(
        self,
        algorithm: AbsAlgorithm,
        memory_capacity: int,
        random_overwrite: bool = False,
        sampler_cls=UniformSampler,
        sampler_kwargs: dict = {}
    ):
        super().__init__()
        self.algorithm = algorithm
        self.experience_memory = ExperienceMemory(memory_capacity, random_overwrite=random_overwrite)
        self.sampler = sampler_cls(self.experience_memory, **sampler_kwargs)

        self.exploring = False

    def choose_action(self, state):
        return self.algorithm.choose_action(state, explore=self.exploring)

    def memorize(self, exp: ExperienceSet) -> bool:
        """
        Store incoming experiences and update if necessary.
        """
        indexes = self.experience_memory.put(exp)
        self.sampler.on_new(exp, indexes)

    def get_batch(self):
        self.sampler.get()

    def reset_memory(self):
        """Clear the experience store."""
        self.experience_memory.clear()

    def update(self):
        batch = self.sampler.get()
        loss_info, indexes = self.algorithm.learn(batch, inplace=True)
        self.sampler.update(indexes, loss_info)

    def explore(self):
        self.exploring = True

    def exploit(self):
        self.exploring = False

    def exploration_step(self):
        if self.algorithm.exploration:
            self.algorithm.exploration.step()
