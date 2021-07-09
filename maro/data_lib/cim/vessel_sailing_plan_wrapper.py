# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .entities import CimBaseDataCollection, Stop
from .vessel_future_stops_prediction import VesselFutureStopsPrediction


class VesselSailingPlanWrapper(VesselFutureStopsPrediction):
    """Wrapper to get vessel sailing plan, this method will return a stop
    list that within configured time peroid (means no same port in list).

    Examples:

        .. code-block:: python

            # Get sailing plan for vessel 0.
            stops = data_cntr.vessel_planned_stops[0]
    """

    def __init__(self, data: CimBaseDataCollection):
        super().__init__(data)

    def __getitem__(self, key):
        assert type(key) == tuple or type(key) == list
        assert len(key) == 3

        vessel_idx = key[0]
        route_idx = key[1]
        next_loc_idx = key[2]

        route_length = len(self._routes[route_idx])

        last_stop: Stop = self._stops[vessel_idx][next_loc_idx]
        last_port_idx = last_stop.port_idx
        last_port_arrival_tick = last_stop.arrival_tick

        stops = self._predict_future_stops(
            vessel_idx, last_port_idx, last_port_arrival_tick, route_length)

        return [(stop.port_idx, stop.arrival_tick) for stop in stops]
