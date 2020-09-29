# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .station import Station


class StationReward:
    """Class to calculate station reward.

    Args:
        stations (list): List of current stations.
        options (dict): Options from configuration file.
    """

    def __init__(self, stations: list, options: dict):
        self._stations = stations
        self._fulfillment_factor = options["fulfillment_factor"]
        self._shortage_factor = options["shortage_factor"]
        self._transfer_cost_factor = options["transfer_cost_factor"]

    def reward(self, station_idx: int) -> float:
        """Get reward for specified station.

        Args:
            station_idx (int): Target station index.

        Returns:
            float: Reward of target station.
        """
        station: Station = self._stations[station_idx]

        reward = self._fulfillment_factor * station.fulfillment \
            - self._shortage_factor * station.shortage \
            - self._transfer_cost_factor * station.transfer_cost

        return reward
