# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import deque
from typing import Deque, List, Tuple

import pandas as pd

from maro.simulator.scenarios.oncall_routing.order import GLOBAL_ORDER_COUNTER, Order


class OncallOrderGenerator(object):
    def __init__(self) -> None:
        super(OncallOrderGenerator, self).__init__()

        self._queue: Deque[Tuple[int, Order]] = deque()

    def get_oncall_orders(self, tick: int) -> List[Order]:
        orders = []
        while len(self._queue) > 0 and self._queue[0][0] == tick:
            _, order = self._queue.popleft()
            orders.append(order)
        return orders


class FromCSVOncallOrderGenerator(OncallOrderGenerator):
    def __init__(self, csv_path: str) -> None:
        super(FromCSVOncallOrderGenerator, self).__init__()

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
        self._queue = deque(buff)


if __name__ == "__main__":
    oncall_order_generator = FromCSVOncallOrderGenerator(
        "C:/workspace/maro/maro/simulator/scenarios/oncall_routing/topologies/example/oncall_orders.csv"
    )
