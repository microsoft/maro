from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable


class AbsMovingAverage(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def record(self, data: object):
        """Record the historical data for forecasting.

        Args:
            data (object): The historical data to record. The type is determined by the need.
        """
        pass

    @abstractmethod
    def forecast(self) -> object:
        """Finish the required forecasting and return the forecating results.

        Returns:
            object: The forecating results. The type is determined by the need.
        """
        pass

    @abstractmethod
    def reset(self):
        pass


class OneStepFixWindowMA(AbsMovingAverage):
    def __init__(self, window_size: int):
        self._count = 0
        self._sum = 0
        self._window_size = window_size
        self._data = deque([0] * self._window_size, maxlen=self._window_size)

    def _record_item(self, item: object):
        # Update the data counter and the sum.
        if self._count == self._window_size:
            self._sum -= self._data[-1]
        else:
            self._count += 1
        self._sum += item

        # Append the item.
        self._data.appendleft(item)

    def record(self, data: object):
        """Record the historical data inside the forecaster.

        Args:
            data (object): It can be a single value of a List of value.
                If a List is given, the data should be sorted from the oldest to the newest.
        """
        if data is None:
            return
        elif not isinstance(data, Iterable):
            data = [data]

        for item in data:
            self._record_item(item=item)

    def forecast(self):
        prediction = self._sum / max(self._count, 1)
        return prediction

    def reset(self):
        self._count = 0
        self._sum
        self._data = deque([0] * self._window_size, maxlen=self._window_size)
