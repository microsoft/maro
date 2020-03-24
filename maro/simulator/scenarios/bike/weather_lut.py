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
        self.type = WeatherType
        self.avg_temp = avg_temp


weather_type = np.dtype(
    [
        ("date", "datetime64[s]"),
        ("weather", "b"),
        ("temp", "f")
    ]
)

def read_weather(file_path: str, start_date_str: str):
    """Read the weather file and return a loopup table with day (tick%24) as key"""
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')

    assert os.path.exists(file_path)

    arr = np.load(file_path)

    weather_lut = {}

    for item in arr[arr["date"]>=start_date]:
        cur_date = item["date"].astype(datetime.datetime)
        delta = cur_date - start_date
        weather_lut[delta.days] = Weather(item["weather"], item["temp"])

    return weather_lut

