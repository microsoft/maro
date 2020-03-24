import datetime

import numpy as np
from dateutil.relativedelta import relativedelta

from .common import Trip

bike_dtype = np.dtype([
    ("start_time", "datetime64[s]"), # datetime
    ("start_cell", "i2"), # id
    ("end_cell", "i2"), # id
    ("duration", "i2"), # min
    ("gendor", "b"), 
    ("usertype", "b"), 
])

class BikeTripReader:
    """Reader"""
    def __init__(self, path: str, start_date: str, max_tick: int, cell_map:dict):
        self._index = 0
        self._cell_map = cell_map
        self._start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        self._max_tick = max_tick # hour
        self._end_date = self._start_date + relativedelta(hours=max_tick)
        self._arr = np.memmap(path, dtype=bike_dtype, mode="c")

        start_filter = self._arr["start_time"] >= self._start_date
        end_filter = self._arr["start_time"] <= self._end_date
        
        self._data_view = self._arr[start_filter & end_filter]
        self._total_items = len(self._data_view)

        self.reset()

    def reset(self):
        """Reset the reader to read from beginning"""
        
        self._index = 0

    def get_trips(self, internal_tick: int):
        """get next event of specified internal_tick, return [] if not exist"""
        trips = []

        # start time of current tick
        start = self._start_date + relativedelta(minutes=internal_tick)

        # next minute
        end = start + relativedelta(minutes=1)

        while self._index < self._total_items:
            item = self._data_view[self._index]
            item_time = item["start_time"]

            if item_time >= start and item_time < end:
                # an valid item
                start_cell_idx = self._cell_map[item["start_cell"]]
                end_cell_idx = self._cell_map[item["end_cell"]]
                end_tick = internal_tick + item["duration"]

                trip = Trip(item_time.astype(datetime.datetime), start_cell_idx, end_cell_idx, end_tick)
                
                trips.append(trip)

                self._index += 1
            else:
                break
            
        return trips
