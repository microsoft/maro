
from abc import ABC, abstractmethod


class LogicBase(ABC):
    # Entity of current logic.
    entity = None

    # Data model instance of current entity.
    data = None

    # Current world.
    world = None

    facility = None

    @abstractmethod
    def initialize(self, config):
        pass

    @abstractmethod
    def step(self, tick: int):
        pass

    @abstractmethod
    def get_metrics(self):
        pass

    @abstractmethod
    def reset(self):
        pass
