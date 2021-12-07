# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from abc import abstractmethod
from typing import Dict, List, Tuple

import pandas as pd
from yaml import safe_load

from maro.simulator.scenarios.oncall_routing import GLOBAL_ORDER_COUNTER, PLAN_RAND_KEY, Coordinate, Order, PlanElement
from maro.simulator.utils import random
from maro.utils import DottableDict


def _load_plan_simple(csv_path: str, start_tick: int, end_tick: int) -> Dict[str, List[PlanElement]]:
    print(f"Loading routes data from {csv_path}.")
    df = pd.read_csv(csv_path, sep=',')

    route_names = sorted(list(set(df["ROUTENBR"])))
    plan_by_route = {}
    for route_name in route_names:
        data = df[df["ROUTENBR"] == route_name]
        data.sort_values(by=['STOPTIME'])

        plan = []
        for e in data.to_dict(orient='records'):
            # TODO
            order = Order(
                order_id=str(next(GLOBAL_ORDER_COUNTER)),
                coordinate=Coordinate(e["LAT"], e["LNG"]),
                open_time=start_tick,
                close_time=end_tick,
                is_delivery=e["IS_DELIVERY"]
            )

            plan.append(PlanElement(order=order))
        plan_by_route[str(route_name)] = plan

    print(f"Loading finished. Loaded data of {len(plan_by_route)} routes.")
    return plan_by_route


def _deprecated_load_sample_length(path: str) -> Dict[str, List[int]]:
    with open(path) as fp:
        ret = safe_load(fp)
    return ret


def _deprecated_load_sample_coords(path: str) -> Dict[str, Tuple[List[Coordinate], List[float]]]:
    ret = {}
    with open(path) as fin:
        for line in fin:
            route_name, coords, probs = line.strip().split("\t")

            new_coords = []
            for elem in coords.split("/"):
                lat, lng = elem.split(",")
                new_coords.append(Coordinate(float(lat), float(lng)))
            probs = [float(elem) for elem in probs.split("/")]

            ret[route_name] = (new_coords, probs)
    return ret


class PlanLoader(object):
    def __init__(self) -> None:
        super(PlanLoader, self).__init__()

    def generate_plan(self) -> Dict[str, List[PlanElement]]:
        return self._generate_plan_impl()

    @abstractmethod
    def _generate_plan_impl(self) -> Dict[str, List[PlanElement]]:
        raise NotImplementedError


class FromHistoryPlanLoader(PlanLoader):
    def __init__(self, csv_path: str, config: DottableDict) -> None:
        super(FromHistoryPlanLoader, self).__init__()
        self._plan = _load_plan_simple(csv_path, config.start_tick, config.end_tick)

    def _generate_plan_impl(self) -> Dict[str, List[PlanElement]]:
        return self._plan


class DeprecatedSamplePlanLoader(PlanLoader):
    def __init__(self, sample_config_path: str, config: DottableDict, pickup_ratio: float = 0.05) -> None:
        super(DeprecatedSamplePlanLoader, self).__init__()

        assert 0.0 < pickup_ratio < 1.0

        self._sample_length = _deprecated_load_sample_length(os.path.join(sample_config_path, "route_length.yml"))
        self._sample_coords = _deprecated_load_sample_coords(os.path.join(sample_config_path, "route_coord.txt"))
        self._route_names = sorted(list(self._sample_coords.keys()))
        self._pickup_ratio = pickup_ratio

        self._start_tick = config.start_tick
        self._end_tick = config.end_tick

    def _generate_plan_impl(self) -> Dict[str, List[PlanElement]]:
        ret = {}
        for route_name in self._route_names:
            # Sample route length. Skip empty routes.
            length = random[PLAN_RAND_KEY].choice(self._sample_length[route_name])
            if length == 0:
                continue

            # Sample coordinates, with weights, with replacement.
            coords = random[PLAN_RAND_KEY].choices(
                population=self._sample_coords[route_name][0],
                weights=self._sample_coords[route_name][1],
                k=length
            )

            # Build plan
            plan = []
            for coord in coords:
                order = Order(
                    order_id=str(next(GLOBAL_ORDER_COUNTER)),
                    coordinate=coord,
                    open_time=self._start_tick,
                    close_time=self._end_tick
                )
                if random[PLAN_RAND_KEY].uniform(0.0, 1.0) < self._pickup_ratio:  # Pickup order
                    # TODO: sample open_time and close_time
                    order.is_delivery = False
                else:  # Delivery order
                    # TODO: sample open_time and close_time
                    order.is_delivery = True
                plan.append(PlanElement(order=order))

            ret[str(route_name)] = plan

        return ret


def get_data_loader(config_path: str, config: DottableDict) -> PlanLoader:
    if config.data_loader_type == "history":
        return FromHistoryPlanLoader(os.path.join(config_path, "routes.csv"), config)
    elif config.data_loader_type == "sample":
        return DeprecatedSamplePlanLoader(config_path, config)
    else:
        raise ValueError("Cannot found correct route data.")
