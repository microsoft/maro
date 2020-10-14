# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from math import ceil

from .entities import CimDataCollection, Stop


class VesselFutureStopsPrediction:
    """Wrapper to get (predict, without noise) vessel future stops, the number of stops is limited by configuration.

    Examples:

        .. code-block:: python

            # Get future stops of vessel 0.
            stops = data_cntr.vessel_future_stops[0]
    """

    def __init__(self, data: CimDataCollection):
        self._vessels = data.vessels_settings
        self._stops = data.vessels_stops
        self._routes = data.routes
        self._route_mapping = data.route_mapping
        self._port_mapping = data.port_mapping
        self._stop_number = data.future_stop_number

    def __getitem__(self, key):
        """Used to support querying future stops by vessel index, last location index, next location index."""
        assert type(key) == tuple or type(key) == list
        assert len(key) == 3

        vessel_idx = key[0]
        last_loc_idx = key[1]
        next_loc_idx = key[2]

        # ignore current port if parking
        start = next_loc_idx + (1 if last_loc_idx == next_loc_idx else 0)

        if last_loc_idx != next_loc_idx:
            start = start - 1

        last_stop = self._stops[vessel_idx][start]
        last_port_idx = last_stop.port_idx
        last_port_arrive_tick = last_stop.arrive_tick

        return self._predict_future_stops(vessel_idx, last_port_idx, last_port_arrive_tick, self._stop_number)

    def _predict_future_stops(self, vessel_idx: int, last_port_idx: int, last_port_arrive_tick: int, stop_number: int):
        """Do predict future stops.
        """
        vessel = self._vessels[vessel_idx]
        speed = vessel.sailing_speed
        duration = vessel.parking_duration
        route_name = vessel.route_name
        route_points = self._routes[self._route_mapping[route_name]]
        route_length = len(route_points)

        last_loc_idx = -1

        # try to find the last location index from route
        for loc_idx, route_point in enumerate(route_points):
            if self._port_mapping[route_point.port_name] == last_port_idx:
                last_loc_idx = loc_idx
                break

        # return if not in current route
        if last_loc_idx < 0:
            return []

        predicted_future_stops = []
        arrive_tick = last_port_arrive_tick

        # predict from configured sailing plan, not from stops
        for loc_idx in range(last_loc_idx + 1, last_loc_idx + stop_number + 1):
            route_info = route_points[loc_idx % route_length]
            port_idx, distance = self._port_mapping[route_info.port_name], route_info.distance

            # NO noise for speed
            arrive_tick += duration + ceil(distance / speed)

            predicted_future_stops.append(
                Stop(-1,  # predict stop do not have valid index
                     arrive_tick,
                     arrive_tick + duration,
                     port_idx,
                     vessel_idx
                     )
            )

        return predicted_future_stops
