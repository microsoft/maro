# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .entities import CimDataCollection


class VesselStopsWrapper:
    """Accessor for vessel stops.

    Examples:

        .. code-block:: python

            # get a stop detail by vessel and location (stop) index
            stop = data_cntr.vessel_stops[vessel_idx, loc_idx]

            # get stop list of a vessel
            stop_list = data_cntr.vessel_stops[vessel_idx]

            # get all stops, NOTE: slice without parameters
            stops = data_cntr.vessel_stops[:]
    """

    def __init__(self, data: CimDataCollection):
        self._stops = data.vessels_stops

    def __getitem__(self, key):
        key_type = type(key)

        if key_type == int:
            # get stops for vessel
            vessel_idx = key
            return self._stops[vessel_idx]
        elif key_type == tuple:
            vessel_idx = key[0]
            loc_idx = key[1]

            return self._stops[vessel_idx][loc_idx]
        elif key_type == slice and key.start is None and key.step is None and key.stop is None:
            return self._stops

        return None
