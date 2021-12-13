# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from abc import abstractmethod
from collections import deque
from typing import Deque, List, Tuple

import pandas as pd
from yaml import safe_load

from maro.simulator.scenarios.oncall_routing import ONCALL_RAND_KEY, Coordinate, Order, OrderIdGenerator
from maro.simulator.utils import random
from maro.utils import DottableDict

from .utils import convert_time_format


class OncallOrderGenerator(object):
    def __init__(self) -> None:
        super(OncallOrderGenerator, self).__init__()

        self._queue: Deque[Tuple[int, Order]] = deque()
        self._id_counter = OrderIdGenerator(prefix="oncall")

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def get_oncall_orders(self, tick: int) -> List[Order]:
        orders = []
        while len(self._queue) > 0 and self._queue[0][0] < tick:
            self._queue.popleft()

        while len(self._queue) > 0 and self._queue[0][0] == tick:
            _, order = self._queue.popleft()
            orders.append(order)
        return orders


class FromHistoryOncallOrderGenerator(OncallOrderGenerator):
    def __init__(self, csv_path: str, coordinate_keep_digit: int) -> None:
        super(FromHistoryOncallOrderGenerator, self).__init__()

        df = pd.read_csv(csv_path, sep=',')
        buff = []
        for e in df.to_dict(orient='records'):
            lat = round(e["LAT"], coordinate_keep_digit)
            lng = round(e["LNG"], coordinate_keep_digit)
            order = Order(
                order_id=self._id_counter.next(),
                coordinate=Coordinate(lat, lng),
                open_time=convert_time_format(e["READYTIME"]),
                close_time=convert_time_format(e["CLOSETIME"]),
                is_delivery=False
            )

            create_time = max(0, order.open_time - random[ONCALL_RAND_KEY].uniform(30, 120))
            buff.append((create_time, order))

        buff.sort(key=lambda x: x[0])
        self._origin_data = buff

    def reset(self) -> None:
        self._queue = deque(self._origin_data)


def normalize_weights(weights: List[float]) -> List[float]:
    weight_sum = sum(weights)
    return [weight / weight_sum for weight in weights]


class SampleOncallOrderGenerator(OncallOrderGenerator):
    def __init__(self, config_path: str, data_loader_config: DottableDict) -> None:
        super(SampleOncallOrderGenerator, self).__init__()

        with open(os.path.join(config_path, "oncall_info.yml")) as fp:
            oncall_info = safe_load(fp)
            self._oncall_numbers = oncall_info["oncall_numbers"]
            self._coords = oncall_info["coordinates"]
            self._open_times = oncall_info["open_times"]
            self._time_windows = oncall_info["time_windows"]
            self._additional_info = oncall_info["additional_info"]

            self._open_times[0] = [convert_time_format(val) for val in self._open_times[0]]

        self._start_tick = data_loader_config.start_tick
        self._end_tick = data_loader_config.end_tick
        self._coordinate_keep_digit = data_loader_config.coordinate_keep_digit

        new_open_times = [[], []]
        for t, weight in zip(self._open_times[0], self._open_times[1]):
            if self._start_tick <= t < self._end_tick - 10:  # TODO
                new_open_times[0].append(t)
                new_open_times[1].append(weight)
        self._open_times = [new_open_times[0], normalize_weights(new_open_times[1])]

    def reset(self) -> None:
        self._id_counter.reset()

        n = random[ONCALL_RAND_KEY].choice(self._oncall_numbers)

        coords = random[ONCALL_RAND_KEY].choices(self._coords[0], weights=self._coords[1], k=n)
        open_times = random[ONCALL_RAND_KEY].choices(self._open_times[0], weights=self._open_times[1], k=n)

        windows = random[ONCALL_RAND_KEY].choices(self._time_windows[0], weights=self._time_windows[1], k=n)
        windows = [max(10, w) for w in windows]

        close_times = [min(self._end_tick, open_times[i] + windows[i]) for i in range(n)]

        buff = []
        for i in range(n):
            lat = round(coords[i][0], self._coordinate_keep_digit)
            lng = round(coords[i][1], self._coordinate_keep_digit)
            order = Order(
                order_id=self._id_counter.next(),
                coordinate=Coordinate(lat, lng),
                open_time=open_times[i],
                close_time=close_times[i],
                is_delivery=False,
            )
            create_time = max(0, order.open_time - int(random[ONCALL_RAND_KEY].uniform(30, 120)))
            buff.append((create_time, order))

        buff.sort(key=lambda x: x[0])
        self._queue = deque(buff)


def get_oncall_generator(config_path: str, data_loader_config: DottableDict) -> OncallOrderGenerator:
    if data_loader_config.oncall_generator_type == "history":
        return FromHistoryOncallOrderGenerator(
            os.path.join(config_path, "oncall_orders.csv"),
            data_loader_config.coordinate_keep_digit
        )
    elif data_loader_config.oncall_generator_type == "sample":
        return SampleOncallOrderGenerator(config_path, data_loader_config)
    else:
        raise ValueError("Cannot found correct oncall data.")
