

class Order:
    def __init__(self, start: int, end: int, num: int, duration: int):
        self._start_station_idx = start
        self._end_station_idx = end
        self._num = num
        self._duration = duration

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
    def duration(self):
        return self._duration

class BikeReturnPayload:
    def __init__(self, tick: int, station_idx: int, num: int):
        self.tick = tick
        self.station_idx = station_idx
        self.num = num