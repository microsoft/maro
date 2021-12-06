# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from abc import abstractmethod
from collections import deque
from typing import Deque, List, Tuple

import pandas as pd
from yaml import safe_load

from maro.simulator.scenarios.oncall_routing import GLOBAL_ORDER_COUNTER, ONCALL_RAND_KEY, Coordinate, Order
from maro.simulator.utils import random
from maro.utils import DottableDict


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


def convert_time_format(t: int) -> int:
    return (t // 100) * 60 + (t % 100)


class FromHistoryOncallOrderGenerator(OncallOrderGenerator):
    def __init__(self, csv_path: str) -> None:
        super(FromHistoryOncallOrderGenerator, self).__init__()

        df = pd.read_csv(csv_path, sep=',')
        buff = []
        for e in df.to_dict(orient='records'):
            order = Order(
                order_id=str(next(GLOBAL_ORDER_COUNTER)),
                coordinate=Coordinate(e["LAT"], e["LNG"]),
                open_time=convert_time_format(e["READYTIME"]),
                close_time=convert_time_format(e["CLOSETIME"])
            )

            create_time = max(0, order.open_time - random[ONCALL_RAND_KEY].uniform(30, 120))
            buff.append((create_time, order))

        buff.sort(key=lambda x: x[0])
        self._origin_data = buff

    def reset(self) -> None:
        self._queue = deque(self._origin_data)


class SampleOncallOrderGenerator(OncallOrderGenerator):
    def __init__(self, config_path: str, config: DottableDict) -> None:
        super(SampleOncallOrderGenerator, self).__init__()

        with open(os.path.join(config_path, "oncall_info.yml")) as fp:
            oncall_info = safe_load(fp)
            self._oncall_numbers = oncall_info["oncall_numbers"]
            self._coords = oncall_info["coordinates"]
            self._open_times = oncall_info["open_times"]
            self._time_windows = oncall_info["time_windows"]
            self._additional_info = oncall_info["additional_info"]

            self._open_times[0] = [convert_time_format(val) for val in self._open_times[0]]

        self._end_tick = config.end_tick

    def reset(self) -> None:
        n = random[ONCALL_RAND_KEY].choice(self._oncall_numbers)

        coords = random[ONCALL_RAND_KEY].choices(self._coords[0], weights=self._coords[1], k=n)
        open_times = random[ONCALL_RAND_KEY].choices(self._open_times[0], weights=self._open_times[1], k=n)
        windows = random[ONCALL_RAND_KEY].choices(self._time_windows[0], weights=self._time_windows[1], k=n)
        close_times = [min(self._end_tick, open_times[i] + windows[i]) for i in range(n)]

        buff = []
        for i in range(n):
            order = Order(
                order_id=str(next(GLOBAL_ORDER_COUNTER)),
                coordinate=coords[i],
                open_time=open_times[i],
                close_time=close_times[i]
            )
            create_time = max(0, order.open_time - random[ONCALL_RAND_KEY].uniform(30, 120))
            buff.append((create_time, order))

        buff.sort(key=lambda x: x[0])
        self._queue = deque(buff)


def get_oncall_generator(config_path: str, config: DottableDict) -> OncallOrderGenerator:
    if os.path.exists(os.path.join(config_path, "oncall_orders.csv")):
        return FromHistoryOncallOrderGenerator(os.path.join(config_path, "oncall_orders.csv"))
    if os.path.exists(os.path.join(config_path, "oncall_info.yml")):
        return SampleOncallOrderGenerator(config_path, config)
    raise ValueError("Cannot found correct oncall data.")
