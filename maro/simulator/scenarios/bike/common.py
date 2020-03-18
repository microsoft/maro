

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
    def __init__(self, tick: int, station_idx: int, num: int):
        self.tick = tick
        self.station_idx = station_idx
        self.num = num