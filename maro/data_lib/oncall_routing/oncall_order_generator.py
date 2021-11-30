# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from abc import abstractmethod
from collections import deque
from typing import Deque, List, Tuple

import pandas as pd
from yaml import safe_load

from maro.simulator.scenarios.oncall_routing import GLOBAL_ORDER_COUNTER, Order
from maro.simulator.scenarios.oncall_routing import ONCALL_RAND_KEY
from maro.simulator.utils import random


class OncallOrderGenerator(object):
    def __init__(self) -> None:
        super(OncallOrderGenerator, self).__init__()

        self._queue: Deque[Tuple[int, Order]] = deque()

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def get_oncall_orders(self, tick: int) -> List[Order]:
        orders = []
        while len(self._queue) > 0 and self._queue[0][0] == tick:
            _, order = self._queue.popleft()
            orders.append(order)
        return orders


class FromHistoryOncallOrderGenerator(OncallOrderGenerator):
    def __init__(self, csv_path: str) -> None:
        super(FromHistoryOncallOrderGenerator, self).__init__()

        df = pd.read_csv(csv_path, sep=',')
        buff = []
        for e in df.to_dict(orient='records'):
            order = Order()
            order.id = next(GLOBAL_ORDER_COUNTER)
            order.coord = (e["LAT"], e["LNG"])
            order.open_time = e["READYTIME"]
            order.close_time = e["CLOSETIME"]

            buff.append((int(order.open_time) // 3, order))  # TODO: fake

        buff.sort(key=lambda x: x[0])
        self._origin_data = buff

    def reset(self) -> None:
        self._queue = deque(self._origin_data)


def add_time(start_time: int, window: int) -> int:
    h, m = start_time // 100, start_time % 100
    m += window
    h += m // 60
    m %= 60
    return h * 100 + m


class SampleOncallOrderGenerator(OncallOrderGenerator):
    def __init__(self, config_path: str) -> None:
        super(SampleOncallOrderGenerator, self).__init__()

        with open(os.path.join(config_path, "oncall_info.yml")) as fp:
            oncall_info = safe_load(fp)
            self._oncall_numbers = oncall_info["oncall_numbers"]
            self._coords = oncall_info["coordinates"]
            self._ready_times = oncall_info["ready_times"]
            self._time_windows = oncall_info["time_windows"]

    def reset(self) -> None:
        n = random[ONCALL_RAND_KEY].choice(self._oncall_numbers)

        coords = random[ONCALL_RAND_KEY].choices(self._coords[0], weights=self._coords[1], k=n)
        open_times = random[ONCALL_RAND_KEY].choices(self._ready_times[0], weights=self._ready_times[1], k=n)
        windows = random[ONCALL_RAND_KEY].choices(self._time_windows[0], weights=self._time_windows[1], k=n)
        close_times = [min(2359, open_times[i] + windows[i]) for i in range(n)]

        self._queue = deque()
        for i in range(n):
            order = Order()
            order.id = next(GLOBAL_ORDER_COUNTER)
            order.coord = coords[i]
            order.open_time = open_times[i]
            order.close_time = close_times[i]
            self._queue.append((int(order.open_time) // 3, order))  # TODO: fake


if __name__ == "__main__":
    oncall_order_generator = FromHistoryOncallOrderGenerator(
        "/maro/simulator/scenarios/oncall_routing/topologies/example_history/oncall_orders.csv"
    )
