from .abs_decisionstrategy import AbsDecisionStrategy


class BikePercentDecisionStrategy(AbsDecisionStrategy):
    """stations will trigger a decision event when its available bikes touch a percentage bar"""
    def __init__(self, stations: list, options: dict):
        super().__init__(stations, options)

        # this strategy need a percentage config
        self._threshold = options["threshold"]


    def get_stations_need_decision(self, tick: int, unit_tick: int) -> list:
        ret = []

        for station in self._stations:
            if station.inventory/station.capacity <= self._threshold:
                ret.append(station.index)

        return ret