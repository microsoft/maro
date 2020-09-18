# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.s

from datetime import date

from maro.data_lib.binary_reader import BinaryReader

from maro.simulator.scenarios.helpers import utc_timestamp_to_timezone

class WeatherTable:
    """Value look-up table, date->weather info"""
    def __init__(self, file: str, timezone):
        self._setup_in_memory_table(file, timezone)

    def _setup_in_memory_table(self, file: str, timezone):
        reader = BinaryReader(file_path=file)

        self._weather_lut = {}

        for item in reader.items(): # just get all the items without filters
            dt = utc_timestamp_to_timezone(item.timestamp, timezone)

            self._weather_lut[dt.date()] = item

    def __getitem__(self, key: date):
        assert type(key) == date
        
        return self._weather_lut.get(key, None)