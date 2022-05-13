# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .entities import CimBaseDataCollection
from .utils import extract_key_of_three_ints
from .vessel_future_stops_prediction import VesselFutureStopsPrediction


class VesselSailingPlanWrapper(VesselFutureStopsPrediction):
    """Wrapper to get vessel sailing plan, this method will return a stop
    list that within configured time period (means no same port in list).

    Examples:

        .. code-block:: python

            # Get sailing plan for vessel 0.
            stops = data_cntr.vessel_planned_stops[0]
    """

    def __init__(self, data: CimBaseDataCollection) -> None:
        super().__init__(data)

    def __getitem__(self, key):
        vessel_idx, route_idx, next_loc_idx = extract_key_of_three_ints(key)
        route_length = len(self._routes[route_idx])
        stops = self._predict_future_stops(vessel_idx, next_loc_idx, route_length)
        return [(stop.port_idx, stop.arrival_tick) for stop in stops]
