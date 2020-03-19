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
        self._total_items = len(self._data_view)

        self.reset()

    def reset(self):
        """Reset the reader to read from beginning"""
        
        self._index = 0

    def get_orders(self, tick: int):
        """get next event of specified tick(in hour), return None if not exist"""
        ret = []

        start = self._start_date + relativedelta(minutes=tick)
        end = start + relativedelta(minutes=1)

        # TODO: need to compare the performance between numpy filter and a simple loop for small batch of data
        # rows = self._data_view[(self._data_view['start_time'] >= start) & (self._data_view['start_time'] < end)]

        # for row in rows:
        #     order = Order(self._station_map[row["start_station"]], self._station_map[row["end_station"]], row["duration"])

        #     ret.append(order)

        while self._index < self._total_items:
            item = self._data_view[self._index]
            item_time = item["start_time"]

            if item_time >= start and item_time < end:
                # an valid item
                start_station_idx = self._station_map[item["start_station"]]
                end_station_idx = self._station_map[item["end_station"]]
                end_tick = tick + item["duration"]

                order = Order(item_time.astype(datetime.datetime), start_station_idx, end_station_idx, end_tick)

                ret.append(order)

                self._index += 1
            else:
                break
            
        return ret
