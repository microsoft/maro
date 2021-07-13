# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .entities import CimBaseDataCollection


class VesselReachableStopsWrapper:
    """Wrapper to get a list of tuple which contains port index and arrive tick in vessel's route.

    Examples:

        .. code-block:: python

            # Get reachable_stops for vessel 0.
            stop_list = data_cntr.reachable_stops[0]
    """

    def __init__(self, data: CimBaseDataCollection):
        self._routes = data.routes
        self._stops = data.vessel_stops

    def __getitem__(self, key):
        assert type(key) == tuple or type(key) == list
        assert len(key) == 3

        vessel_idx = key[0]
        route_idx = key[1]
        next_loc_idx = key[2]

        route_length = len(self._routes[route_idx])
        stops = self._stops[vessel_idx][
            next_loc_idx + 1: next_loc_idx + 1 + route_length]

        return [(stop.port_idx, stop.arrival_tick) for stop in stops]
