# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from abc import abstractmethod
from typing import Dict, List

import pandas as pd

from maro.simulator.scenarios.oncall_routing import Coordinate, Order, OrderIdGenerator
from maro.simulator.scenarios.oncall_routing.coordinate import CoordinateClipper
from maro.utils import clone, DottableDict

from .utils import convert_time_format


def _load_orders_by_route_simple(
    csv_path: str,
    start_tick: int,
    end_tick: int,
    id_counter: OrderIdGenerator,
    coord_clipper: CoordinateClipper
) -> Dict[str, List[Order]]:
    print(f"Loading routes data from {csv_path}.")
    df = pd.read_csv(csv_path, sep=',')

    route_names = sorted(list(set(df["ROUTENBR"])))
    orders_by_route = {}
    for route_name in route_names:
        data = df[df["ROUTENBR"] == route_name]
        data.sort_values(by=['STOPTIME'])

        orders = []
        for e in data.to_dict(orient='records'):
            # TODO
            order = Order(
                order_id=id_counter.next(),
                coordinate=coord_clipper.clip(Coordinate(e["LAT"], e["LNG"])),
                open_time=start_tick if e["IS_DELIVERY"] else convert_time_format(int(e["READYTIME"])),
                close_time=end_tick if e["IS_DELIVERY"] else convert_time_format(int(e["CLOSETIME"])),
                is_delivery=e["IS_DELIVERY"]
            )
            orders.append(order)
        orders_by_route[str(route_name)] = orders

    print(f"Loading finished. Loaded data of {len(orders_by_route)} routes.")
    return orders_by_route


class PlanLoader(object):
    def __init__(self) -> None:
        super(PlanLoader, self).__init__()
        self._id_counter = OrderIdGenerator(prefix="planned")

    def generate_route_orders(self) -> Dict[str, List[Order]]:
        return self._generate_route_orders_impl()

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _generate_route_orders_impl(self) -> Dict[str, List[Order]]:
        raise NotImplementedError


class FromHistoryPlanLoader(PlanLoader):
    def __init__(self, csv_path: str, data_loader_config: DottableDict, coord_clipper: CoordinateClipper) -> None:
        super(FromHistoryPlanLoader, self).__init__()
        self._orders_by_route = _load_orders_by_route_simple(
            csv_path,
            data_loader_config.start_tick,
            data_loader_config.end_tick,
            self._id_counter,
            coord_clipper
        )

    def reset(self) -> None:
        pass

    def _generate_route_orders_impl(self) -> Dict[str, List[Order]]:
        # Return copies of orders
        return {route_name: [clone(order) for order in orders] for route_name, orders in self._orders_by_route.items()}


def get_data_loader(
    config_path: str,
    data_loader_config: DottableDict,
    coord_clipper: CoordinateClipper
) -> PlanLoader:
    if data_loader_config.data_loader_type == "history":
        return FromHistoryPlanLoader(os.path.join(config_path, "routes.csv"), data_loader_config, coord_clipper)
    else:
        raise ValueError("Cannot found correct route data.")
