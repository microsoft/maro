from abc import ABC, abstractmethod
from maro.simulator.frame import SnapshotList, Frame
from maro.simulator.event_buffer import EventBuffer

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
    def snapshot_list(self): 
        pass

    @abstractmethod
    def step(self, tick: int):
        pass

    @abstractmethod
    def post_step(self, tick: int):
        pass

