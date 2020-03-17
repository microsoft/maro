import numpy as np
from .common import Order

bike_dtype = np.dtype([
    ("tick", "i4"),
    ("start_station", "i4"),
    ("end_station", "i4"),
    ("duration", "i4"),
    ("number", "i4")
])

class BikeDataReader:
    def __init__(self, path: str):
        self._index = 0
        self._arr = np.memmap(path, dtype=bike_dtype, mode="c")

    def reset(self):
        """Reset the reader to read from beginning"""

        self._index = 0

    def get_orders(self, tick: int):
        """get next event of specified tick, return None if not exist"""
        ret = []

        while self._index < len(self._arr):
            item = self._arr[self._index]

            if item["tick"] == tick:
                ret.append(item)

                self._index+=1
            else:
                break

        return ret

        # item = self._arr[tick]

        # if item["tick"] == tick:
        #     return item
        # else:
        #     return None