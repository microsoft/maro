
from abc import ABC, abstractmethod


class LogicBase(ABC):
    # configured class name
    data_class: str = None

    # index of the data model index
    data_index: int = None

    # Current world.
    world = None

    facility = None

    configs: dict = None

    id: int = None

    def __init__(self):
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = self.world.get_datamodel(self.data_class, self.data_index)

        return self._data

    def initialize(self, configs: dict):
        self.configs = configs
        self.data.initialize(configs.get("data", {}))

    @abstractmethod
    def step(self, tick: int):
        pass

    @abstractmethod
    def get_metrics(self):
        pass

    def reset(self):
        self.data.reset()

    def set_action(self, action):
        pass
