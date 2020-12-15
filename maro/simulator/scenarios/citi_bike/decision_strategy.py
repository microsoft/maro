# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from math import floor
from typing import Dict

import numpy as np

from maro.simulator.scenarios.matrix_accessor import MatrixAttributeAccessor

from .common import DecisionType, ExtraCostMode
from .station import Station


class DistanceFilter:
    """Filter neighbors by distance (from near to far), and return first N configured number of neighbors.

    Args:
        conf (dict): Configuration of this filter.
        strategy (BikeDecisionStrategy): BikeDecisionStrategy instance to provide information.
    """

    def __init__(self, conf: dict, strategy):
        self._output_num = conf["num"]
        self._strategy = strategy

    def filter(self, station_idx: int, deision_type: DecisionType, source: Dict[int, int]) -> Dict[int, int]:
        """Filter neighbors by distance from neareat to farest.

        Args:
            station_idx (int): Station index to sort its neighbors.
            deision_type (DecisionType): Current decision type.
            source (dict): Input neighbors for sorting.

        Returns:
            dict: N nearest neighbors.
        """
        output_num = min(self._output_num, len(source))

        neighbors = self._strategy._get_neighbors(station_idx)

        output_neighbors = neighbors[0:output_num]

        result = {}

        for neighbor_idx, _ in output_neighbors:
            result[neighbor_idx] = source[neighbor_idx]

        return result

    def reset(self):
        """Reset internal states."""
        pass


class RequirementsFilter:
    """Filters by demand or supply requirement number, from more to less.

    Args:
        conf (dict): Configuration of this filter.
    """

    def __init__(self, conf: dict):
        self._output_num = conf["num"]

    def filter(self, station_idx: int, deision_type: DecisionType, source: Dict[int, int]) -> Dict[int, int]:
        """Filter neighbors by requirements number.

        Args:
            station_idx (int): Station index to sort its neighbors.
            decision_type (DecisionType): Current decision type.
            source (Dict[int, int]): Neighbors to sort.

        Returns:
            dict: N neighors sorted by requirement.
        """
        output_num = min(self._output_num, len(source))

        neighbor_scope = sorted(source.items(),
                                key=lambda kv: (kv[1], kv[0]), reverse=True)

        return {neighbor_scope[i][0]: neighbor_scope[i][1] for i in range(output_num)}

    def reset(self):
        """Reset internal states."""
        pass


class TripsWindowFilter:
    """Filter neighbors by trip requirement in latest N windows.

    NOTE:
        This filter will cache its states, need to be reset before each episode.

    Args:
        conf (dict): Configuration of this fileter.
        snapshot_list (SnapshotList): Snapshots used to quering states.
    """

    def __init__(self, conf: dict, snapshot_list):
        self._output_num = conf["num"]
        self._windows = conf["windows"]
        self._snapshot_list = snapshot_list

        self._window_states_cache = {}

    def filter(self, station_idx: int, decision_type: DecisionType, source: Dict[int, int]) -> Dict[int, int]:
        """Filter neighbors by trip requirements in latest N windows.

        Args:
            station_idx (int): Station index to sort its neighbors.
            decision_type (DecisionType): Current decision type.
            source (Dict[int, int]): Neighbors to sort.

        Returns:
            dict: N neighors sorted by requirement.
        """
        output_num = min(self._output_num, len(source))

        avaiable_frame_indices = self._snapshot_list.get_frame_index_list()

        # max windows we can get
        available_windows = min(self._windows, len(avaiable_frame_indices))

        # get frame index list for latest N windows
        avaiable_frame_indices = avaiable_frame_indices[-available_windows:]

        source_trips = {}

        for i, frame_index in enumerate(avaiable_frame_indices):
            if i == available_windows - 1 or frame_index not in self._window_states_cache:
                # overwrite latest one, since it may be changes, and cache not exist one
                trip_state = self._snapshot_list["stations"][frame_index::"trip_requirement"]

                self._window_states_cache[frame_index] = trip_state

            trip_state = self._window_states_cache[frame_index]

            for neighbor_idx, _ in source.items():
                trip_num = trip_state[neighbor_idx]

                if neighbor_idx not in source_trips:
                    source_trips[neighbor_idx] = trip_num
                else:
                    source_trips[neighbor_idx] += trip_num

        is_sort_reverse = False

        if decision_type == DecisionType.Demand:
            is_sort_reverse = True

        sorted_neighbors = sorted(source_trips.items(), key=lambda kv: (kv[1], kv[0]), reverse=is_sort_reverse)

        result = {}

        for neighbor_idx, _ in sorted_neighbors[0: output_num]:
            result[neighbor_idx] = source[neighbor_idx]

        return result

    def reset(self):
        self._window_states_cache.clear()


class BikeDecisionStrategy:
    """Helper to provide decision related logic.

    Args:
        stations (list): List for current stations.
        distance_adj (list): Distance adj for sorting.
        snapshots (SnapshotList): Snapshots used to query states.
        options (dict): Additional options from configuration file.
    """

    def __init__(self, stations: list, distance_adj: MatrixAttributeAccessor, snapshots, options: dict):
        self._filter_cls_mapping = {
            "distance": {
                "cls": DistanceFilter,
                "options": [self]
            },
            "requirements": {
                "cls": RequirementsFilter
            },
            "trip_window": {
                "cls": TripsWindowFilter,
                "options": [snapshots]
            }
        }

        self._filters = []
        self._stations = stations
        self._distance_adj = distance_adj

        # used to cache the neighbors after queried from frame, as it will not be changed
        self._neighbors_cache = {}

        self.resolution = options["resolution"]
        self.time_mean = options["effective_time_mean"]
        self.supply_water_mark_ratio = options["supply_water_mark_ratio"]
        self.demand_water_mark_ratio = options["demand_water_mark_ratio"]
        self.time_std = options["effective_time_std"]

        action_scope_options = options["action_scope"]

        self.scope_low_ratio = action_scope_options["low"]
        self.scope_high_ratio = action_scope_options["high"]

        self._extra_cost_mode = ExtraCostMode(options["extra_cost_mode"])

        self._construct_action_scope_filters(action_scope_options)

    @property
    def transfer_time(self) -> int:
        """int: Transfer time from one station to another."""
        return round(np.random.normal(self.time_mean, scale=self.time_std))

    def is_decision_tick(self, tick: int) -> bool:
        """If it is time to generate a decision event.

        Args:
            tick (int): Tick to check.

        Returns:
            bool: True if current tick should check deicion, or False.
        """
        return (tick + 1) % self.resolution == 0

    def get_stations_need_decision(self, tick: int) -> list:
        """Get stations that need to take an action from agent at current tick.

        Args:
            tick (int): Tick to check.

        Returns:
            list: List of station indices who need action at this tick.
        """

        stations = []

        if (tick + 1) % self.resolution == 0:
            for station in self._stations:
                cur_ratio = station.bikes / station.capacity

                # if cell has too many available bikes, then we ask an action
                if cur_ratio >= self.supply_water_mark_ratio:
                    stations.append((station.index, DecisionType.Supply))
                elif cur_ratio <= self.demand_water_mark_ratio:
                    stations.append((station.index, DecisionType.Demand))

        return stations

    def action_scope(self, station_idx: int, decision_type: DecisionType):
        """Calculate action scopes for self and N neighbors.

        Args:
            station_idx (int): Target station index.
            decision_type (DecisionType): Current decision type.

        Returns:
            dict : Dictionary that key is station index, value is the max supply/demand value.
        """
        station: Station = self._stations[station_idx]

        neighbors = self._get_neighbors(station_idx)

        neighbor_scope = {}

        for neighbor_idx, _ in neighbors:
            if neighbor_idx >= 0:
                neighbor_station: Station = self._stations[neighbor_idx]

                # we should not transfer bikes to a cell which already meet the high water mark ratio
                if decision_type == DecisionType.Supply:
                    # for supply decision, we provide max bikes that neighbor can accept
                    max_bikes = neighbor_station.capacity - neighbor_station.bikes
                else:
                    # for demand decision, this will be max bikes that neighbor can provide
                    max_bikes = floor(neighbor_station.bikes * self.scope_high_ratio)

                neighbor_scope[neighbor_idx] = max_bikes

        for nb_filter in self._filters:
            neighbor_scope = nb_filter.filter(station_idx, decision_type, neighbor_scope)

        # how many bikes we can supply to other stations from current cell
        if decision_type == DecisionType.Supply:
            neighbor_scope[station_idx] = floor(station.bikes * (1 - self.scope_low_ratio))
        else:
            # how many bike we can accept
            neighbor_scope[station_idx] = station.capacity - station.bikes

        return neighbor_scope

    def move_to_neighbor(self, src_station: Station, cur_station: Station, bike_number: int):
        """Move extra bikes to neighbor when transferred bikes more than available docks.

        Args:
            src_station (Station): Which station current bike come from.
            cur_station (Station): Which station current bike arrive.
            bike_number (int): How many bikes need to move.

        NOTE:
            Since we have a full neighbors list now, we do not need the N-step way to move, just 1 for now,
        we use distance (order index) as factor of extra cost now
        """
        total_cost = 0
        cost = 0

        neighbors = self._get_neighbors(cur_station.index)

        # move to 1-step neighbors
        for order_index, neighbor in enumerate(neighbors):
            neighbor_idx, distance = neighbor

            # ignore source cell and padding cell
            if neighbor_idx < 0:
                continue

            neighbor = self._stations[neighbor_idx]
            neighbor_bikes = neighbor.bikes
            accept_number = neighbor.capacity - neighbor_bikes

            # how many bikes this cell can accept
            accept_number = min(accept_number, bike_number)
            neighbor.bikes = neighbor_bikes + accept_number

            cost = self._calculate_extra_cost(
                accept_number, distance, order_index)

            total_cost += cost

            self._set_extra_cost(src_station, cur_station, neighbor, cost)

            bike_number = bike_number - accept_number

            if bike_number == 0:
                break

        return total_cost

    def reset(self):
        """Reset internal states."""
        for filter_instance in self._filters:
            filter_instance.reset()

    def _construct_action_scope_filters(self, conf: dict):
        for filter_conf in conf["filters"]:
            filter_type = filter_conf["type"]

            filter_mapping = self._filter_cls_mapping[filter_type]

            filter_cls = filter_mapping["cls"]

            options = filter_mapping.get("options", None)

            if options is None:
                filter = filter_cls(filter_conf)
            else:
                filter = filter_cls(filter_conf, *options)

            self._filters.append(filter)

    def _calculate_extra_cost(self, number: int, distance: float, neighbor_index: int):
        # TODO: update it later
        return number * (neighbor_index + 1)

    def _set_extra_cost(self, src_station: Station, target_station: Station, neighbor: Station, cost: int):
        """set extra cost to station according to the mode"""
        if self._extra_cost_mode == ExtraCostMode.Source:
            # extra cost from source cell
            src_station.extra_cost += cost
        elif self._extra_cost_mode == ExtraCostMode.Target:
            target_station.extra_cost += cost
        else:
            neighbor.extra_cost += cost

    def _get_neighbors(self, station_idx: int):
        """get neighbors for station, from cache if exists"""
        # check cache first
        neighbors = self._neighbors_cache.get(station_idx, None)

        if neighbors is None:
            distances = self._distance_adj[station_idx]

            # index is the station index
            neighbors = [(index, dist) for index, dist in enumerate(distances) if dist != 0.0]

            # sort by distance
            neighbors = sorted(neighbors, key=lambda item: item[1])

            self._neighbors_cache[station_idx] = neighbors

        return neighbors
