

class Order:
    def __init__(self, date, start_station: int, end_station: int, end_tick: int, usertype: int = 0, gendor:int = 0, num: int=1):
        self.date = date
        self.start_station = start_station
        self.end_station = end_station
        self.number = num
        self.end_tick = end_tick
        self.usertype = usertype
        self.gendor = gendor

    def __repr__(self):
        return f"(Order start station: {self.start_station}, end station: {self.end_station}, end tick: {self.end_tick})"

class BikeReturnPayload:
    def __init__(self, target_station_idx: int, num: int = 1):
        self.target_station = target_station_idx
        self.number = num

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
            self._action_scope = self._action_scope_func(self.station_idx)

        return self._action_scope

    def __repr__(self):
        return f"DecisionEvent(station: {self.station_idx}, action scope: {self.action_scope})"

class Action:
    def __init__(self, station_idx:int, number: int):
        self.station = station_idx
        self.number = number

    def __repr__(self):
        return f"Action(station: {self.station}, number: {self.number})"