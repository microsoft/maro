from abc import ABC, abstractmethod
from math import ceil

from maro.simulator.event_buffer import EventBuffer
from maro.simulator.frame import Frame, SnapshotList


class AbsSubBusinessEngine(ABC):
    def __init__(self, start_tick: int, max_tick: int, frame_resolution: int, config: dict, event_buffer: EventBuffer):
        self._start_tick = start_tick
        self._max_tick = max_tick
        self._frame_resolution = frame_resolution
        self._config = config
        self._event_buffer = event_buffer
        self._name = config["name"]

    @property
    def name(self):
        return self._name
    
    @property
    @abstractmethod
    def finance_type(self):
        pass

    @property
    @abstractmethod
    def frame(self):
        pass

    @property
    @abstractmethod
    def snapshot_list(self): 
        pass

    @property
    @abstractmethod
    def name_mapping(self):
        pass

    @abstractmethod
    def step(self, tick: int):
        pass

    @abstractmethod
    def post_step(self, tick: int):
        pass

    @abstractmethod
    def post_init(self, max_tick):
        pass

    @abstractmethod
    def take_action(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    def max_tick(self):
        return self._max_tick
