# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import date

from maro.data_lib.binary_reader import BinaryReader
from maro.simulator.scenarios.helpers import utc_timestamp_to_timezone


class WeatherTable:
    """Value look-up table, used to query weather information by datetime.

    After initializing, this lut object can be used as a normal dictionary which key is the datetime object.

    .. code-block:: python

        weather_lut = WeatherTable(path, UTC)

        # Get weather at specified datetime, usually day leve.
        weather = weather_lut[your_date]

    Args:
        file (str): Binary file that contains weather information.
        timezone (object): Target timezone, used to convert timestamp from binary.
    """

    def __init__(self, file: str, timezone):
        self._setup_in_memory_table(file, timezone)

    def _setup_in_memory_table(self, file: str, timezone):
        reader = BinaryReader(file_path=file)

        self._weather_lut = {}

        # just get all the items without filters
        for item in reader.items():
            dt = utc_timestamp_to_timezone(item.timestamp, timezone)

            self._weather_lut[dt.date()] = item

    def __getitem__(self, key: date):
        assert type(key) == date

        return self._weather_lut.get(key, None)
