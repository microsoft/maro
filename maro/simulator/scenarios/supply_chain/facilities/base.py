
from abc import ABC, abstractmethod


class FacilityBase(ABC):
    world = None

    storage = None
    distribution = None
    transports = None

    configs: dict = None
    id: int = None

    @abstractmethod
    def step(self, tick: int):
        # called per tick
        pass

    @abstractmethod
    def build(self, configs: dict):
        # called to build components, but without data model instance.
        pass

    @abstractmethod
    def initialize(self):
        # called after data model instance is ready.
        pass

    @abstractmethod
    def reset(self):
        # called per episode.
        pass
