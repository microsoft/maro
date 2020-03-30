import datetime

import numpy as np
from dateutil.relativedelta import relativedelta

from .common import Trip

bike_dtype = np.dtype([
    ("start_time", "datetime64[s]"), # datetime
    ("start_station", "i2"), # id
    ("end_station", "i2"), # id
    ("duration", "i2"), # min
    ("gendor", "b"), 
    ("usertype", "b"), 
    ("start_cell", "i2"),
    ("end_cell", "i2")
])

class BikeTripReader:
    """Reader"""
    def __init__(self, path: str, start_date: str, max_tick: int):
        self._index = 0
        self._max_tick = max_tick #
        self._arr = np.memmap(path, dtype=bike_dtype, mode="c")

        # this will be the tick = 0
        start_date = self._arr[0]["start_time"].astype(datetime.datetime)
        self.start_date = datetime.datetime(start_date.year, start_date.month, start_date.day, start_date.hour, start_date.minute)
        
        start_filter = self._arr["start_time"] >= self.start_date

        self._data_view = None

        if max_tick == -1:
            self._data_view = self._arr[start_filter]
        elif max_tick > 0:
            end_date = self.start_date + relativedelta(minutes=max_tick)
            end_filter = self._arr["start_time"] <= end_date

            self._data_view = self._arr[start_filter & end_filter]
        else:
            raise "Invalid max tick to initialize."

        self._total_items = len(self._data_view)

        self.reset()

    def reset(self):
        """Reset the reader to read from beginning"""
        
        self._index = 0

    def get_trips(self, tick: int):
        """get next event of specified tick, return [] if not exist"""
        trips = []

        # start time of current tick
        start = self.start_date + relativedelta(minutes=tick)

        # next minute
        end = start + relativedelta(minutes=1)

        while self._index < self._total_items:
            item = self._data_view[self._index]
            item_time = item["start_time"]

            if item_time >= start and item_time < end:
                # an valid item
                start_cell_idx = item["start_cell"]
                end_cell_idx = item["end_cell"]
                end_tick = tick + item["duration"]

                trip = Trip(item_time.astype(datetime.datetime), start_cell_idx, end_cell_idx, end_tick)
                
                trips.append(trip)

                self._index += 1
            elif item_time < start:
                # used to filter invalid dataset
                self._index += 1
            else:
                break
            
        return trips
