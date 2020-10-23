from abc import ABC, abstractmethod

from maro.simulator.scenarios.finance.common import Action
from maro.event_buffer import EventBuffer


class AbsSubBusinessEngine(ABC):
    def __init__(
            self, beginning_timestamp: int, start_tick: int, max_tick: int,
            frame_resolution: int, config: dict, event_buffer: EventBuffer
    ):
        self._beginning_timestamp = beginning_timestamp
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
    def account(self):
        pass

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
    def take_action(self, action: Action, remaining_money: float, tick: int):
        pass

    @abstractmethod
    def cancel_order(self, action: Action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    def max_tick(self):
        return self._max_tick
