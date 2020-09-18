# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.s

from .station import Station

class StationReward:
    """Class to calculate station reward"""
    def __init__(self, stations: list, options: dict):
        self._stations = stations
        self._fulfillment_factor = options["fulfillment_factor"]
        self._shortage_factor = options["shortage_factor"]
        self._transfer_cost_factor = options["transfer_cost_factor"]

    def reward(self, station_idx: int):
        """Get reward for specified station"""
        station: Station = self._stations[station_idx]

        reward = self._fulfillment_factor * station.fulfillment \
                - self._shortage_factor * station.shortage \
                - self._transfer_cost_factor * station.transfer_cost

        return reward