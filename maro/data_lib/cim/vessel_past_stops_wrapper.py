# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .entities import CimDataCollection


class VesselPastStopsWrapper:
    """Wrapper to get vessel past stops, it will be padding with None if stops number less than configured one.

    Examples:

        .. code-block:: python

            # Get past stops of vessel 0.
            stops = data_cntr.vessel_past_stops[0]
    """

    def __init__(self, data: CimDataCollection):
        self._stop_number = data.past_stop_number
        self._stops = data.vessels_stops

    def __getitem__(self, key):
        assert type(key) == tuple or type(key) == list
        assert len(key) == 3

        vessel_idx = key[0]
        last_loc_idx = key[1]
        next_loc_idx = key[2]

        # ignore current port if parking
        start = next_loc_idx + (1 if last_loc_idx == next_loc_idx else 0)

        # avoid negative index
        start = max(next_loc_idx - self._stop_number, 0)

        past_stop_list = self._stops[vessel_idx][start: next_loc_idx]

        # padding with None
        for _ in range(self._stop_number - len(past_stop_list)):
            past_stop_list.insert(0, None)

        return past_stop_list
