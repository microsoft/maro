# lookup table for weather info
import os
import datetime
import numpy as np

from enum import IntEnum

class WeatherType(IntEnum):
    SUNNY = 0,
    RAINY = 1,
    SNOWY = 2,
    SLEET = 3


class Weather:
    def __init__(self, type: WeatherType, avg_temp: float):
        self.type = type
        self.avg_temp = avg_temp

    def __repr__(self):
        return f"Weather(type: {self.type}, avg temp: {self.avg_temp})"

# numpy dtype in file
weather_type = np.dtype(
    [
        ("date", "datetime64[s]"),
        ("weather", "b"),
        ("temp", "f")
    ]
)


class WeatherTable:
    """Lookup table to query weather information for a day"""
    def __init__(self, weather_file: str, start_date_str: str):
        assert os.path.exists(weather_file)

        self.start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')

        self._weather_dict = {}

        self._cache_table(weather_file)

    def _cache_table(self, weather_file: str):
        arr = np.load(weather_file)

        # we only keep sub-set of weathers in memory to reduce cost
        for item in arr[arr["date"] >= self.start_date]:
            cur_date = item["date"].astype(datetime.datetime)
            delta = cur_date - self.start_date

            self._weather_dict[delta.days] = Weather(WeatherType(item["weather"]), item["temp"])

        arr = None

    def __len__(self):
        return len(self._weather_dict)

    def __getitem__(self, key):
        if type(key) is datetime.datetime:
            delta = key - self.start_date

            key = delta.days

        return self._weather_dict[key]