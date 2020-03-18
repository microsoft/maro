

class Order:
    def __init__(self, start_station: int, end_station: int, end_tick: int, num: int=1):
        self._start_station_idx = start_station
        self._end_station_idx = end_station
        self._num = num
        self._end_tick = end_tick

    @property
    def start_station(self):
        return self._start_station_idx

    @property
    def end_station(self):
        return self._end_station_idx

    @property
    def number(self):
        return self._num

    @property
    def end_tick(self):
        return self._end_tick

    def __repr__(self):
        return f"(Order start station: {self.start_station}, end station: {self.end_station}, end tick: {self.end_tick})"

class BikeReturnPayload:
    def __init__(self, target_station_idx: int, num: int = 1):
        self._target_station_idx = target_station_idx
        self._num = num

    @property
    def target_station(self) -> int:
        return self._target_station_idx

    @property
    def number(self) -> int:
        return self._num

    def __repr__(self):
        return f"(Bike return payload, target station: {self.target_station}, number: {self.number})"


class DecisionEvent:
    def __init__(self, station_idx: int, tick: int, action_scope_func: callable):
        self.station_idx = station_idx
        self.tick = tick
        self._action_scope = None
        self._action_scope_func = action_scope_func

    @property
    def action_scope(self):
        if self._action_scope is None:
            self._action_scope = self._action_scope_func()

        return self._action_scope_func

class Action:
    def __init__(self, station_idx:int, number: int):
        self._station_idx = station_idx
        self._num = number

    @property
    def station(self):
        return self._station_idx

    @property
    def number(self):
        return self._num