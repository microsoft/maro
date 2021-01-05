# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections import namedtuple
from csv import DictReader

StationInfo = namedtuple("StationInfo", ["index", "bikes", "capacity", "id"])


def get_station_info(station_state_file: str):
    """Get stations information from specified csv file.

    Args:
        station_state_file (str): File path that contains station initial state info.

    Returns:
        list: List of station information.
    """
    stations_info = []
    if station_state_file.startswith("~"):
        station_state_file = os.path.expanduser(station_state_file)
    with open(station_state_file, "r") as fp:
        reader = DictReader(fp)

        for row in reader:
            si = StationInfo(
                int(row["station_index"]),
                int(row["init"]),  # init bikes
                int(row["capacity"]),
                # It's a patch for the uncleaned data in the original files, such as 12345.0
                int(float(row["station_id"]))
            )

            stations_info.append(si)

    return stations_info
