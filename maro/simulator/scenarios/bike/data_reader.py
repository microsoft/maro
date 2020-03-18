import numpy as np
import datetime
from .common import Order
from dateutil.relativedelta import relativedelta

bike_dtype = np.dtype([
    ("start_time", "datetime64[s]"), # datetime
    ("start_station", "i4"), # id
    ("end_station", "i4"), # id
    ("duration", "i4"), # min
    ("gendor", "b"), 
    ("usertype", "b"), 
])

class BikeDataReader:
    def __init__(self, path: str, start_date: str, max_tick: int, station_map:dict):
        self._index = 0
        self._station_map = station_map
        self._start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        self._max_tick = max_tick # hour
        self._end_date = self._start_date + relativedelta(hours=max_tick)
        self._arr = np.memmap(path, dtype=bike_dtype, mode="c")

        start_filter = self._arr["start_time"] >= self._start_date
        end_filter = self._arr["start_time"] >= self._start_date
        self._data_view = self._arr[start_filter & end_filter]


        self.reset()

    def reset(self):
        """Reset the reader to read from beginning"""
        
        self._index = 0

    def get_orders(self, tick: int):
        """get next event of specified tick(in hour), return None if not exist"""
        ret = []

        start = self._start_date + relativedelta(hours=tick)
        end = start + relativedelta(days=1)

        rows = self._data_view[(self._data_view['start_time'] >= start) & (self._data_view['start_time'] < end)]

        for row in rows:
            order = Order(self._station_map[row["start_station"]], self._station_map[row["end_station"]], row["duration"])

            ret.append(order)
            
        return ret
