# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from abc import abstractmethod
from typing import Dict, List, Tuple

import pandas as pd
from yaml import safe_load

from maro.simulator.scenarios.oncall_routing import (
    GLOBAL_ORDER_COUNTER, PLAN_RAND_KEY, Coordinate, Order, PlanElement, RouteNumber
)
from maro.simulator.utils import random

rtb_fake_order = Order()
rtb_fake_order.id = next(GLOBAL_ORDER_COUNTER)
rtb_fake_order.coord = (32.72329226, -117.0718922)


def _load_plan_simple(csv_path: str) -> Dict[RouteNumber, List[PlanElement]]:
    print(f"Loading routes data from {csv_path}.")
    df = pd.read_csv(csv_path, sep=',')

    route_numbers = sorted(list(set(df["ROUTENBR"])))
    plan_by_route = {}
    for route_number in route_numbers:
        data = df[df["ROUTENBR"] == route_number]
        data.sort_values(by=['STOPTIME'])

        plan = []
        for e in data.to_dict(orient='records'):
            # TODO
            order = Order()
            order.id = next(GLOBAL_ORDER_COUNTER)
            order.coord = Coordinate(e["LAT"], e["LNG"])
            order.open_time = e["READYTIME"]
            order.close_time = e["CLOSETIME"]
            order.is_delivery = e["IS_DELIVERY"]

            plan.append(PlanElement(order=order, est_arr_time=-1, act_arr_time=-1))
        plan.append(PlanElement(rtb_fake_order, est_arr_time=-1, act_arr_time=-1))
        plan_by_route[route_number] = plan

    print(f"Loading finished. Loaded data of {len(plan_by_route)} routes.")
    return plan_by_route


class PlanLoader(object):
    def __init__(self) -> None:
        super(PlanLoader, self).__init__()

    def generate_plan(self) -> Dict[RouteNumber, List[PlanElement]]:
        return self._generate_plan_impl()

    @abstractmethod
    def _generate_plan_impl(self) -> Dict[RouteNumber, List[PlanElement]]:
        raise NotImplementedError


class FromHistoryPlanLoader(PlanLoader):
    def __init__(self, csv_path: str) -> None:
        super(FromHistoryPlanLoader, self).__init__()
        self._plan = _load_plan_simple(csv_path)

    def _generate_plan_impl(self) -> Dict[RouteNumber, List[PlanElement]]:
        return self._plan


def _load_sample_length(path: str) -> Dict[int, List[int]]:
    with open(path) as fp:
        ret = safe_load(fp)
    return ret


def _load_sample_coords(path: str) -> Dict[int, Tuple[List[Coordinate], List[float]]]:
    ret = {}
    with open(path) as fin:
        for line in fin:
            route_number, coords, probs = line.strip().split("\t")
            route_number = int(route_number)

            new_coords = []
            for elem in coords.split("/"):
                lat, lng = elem.split(",")
                new_coords.append(Coordinate(float(lat), float(lng)))
            probs = [float(elem) for elem in probs.split("/")]

            ret[route_number] = (new_coords, probs)
    return ret


class SamplePlanLoader(PlanLoader):
    def __init__(self, sample_config_path: str, pickup_ratio: float = 0.05) -> None:
        super(SamplePlanLoader, self).__init__()

        assert 0.0 < pickup_ratio < 1.0

        self._sample_length = _load_sample_length(os.path.join(sample_config_path, "route_length.yml"))
        self._sample_coords = _load_sample_coords(os.path.join(sample_config_path, "route_coord.txt"))
        self._route_numbers = sorted(list(self._sample_coords.keys()))
        self._pickup_ratio = pickup_ratio

    def _generate_plan_impl(self) -> Dict[RouteNumber, List[PlanElement]]:
        ret = {}
        for route_number in self._route_numbers:
            # Sample route length. Skip empty routes.
            length = random[PLAN_RAND_KEY].choice(self._sample_length[route_number])
            if length == 0:
                continue

            # Sample coordinates, with weights, with replacement.
            coords = random[PLAN_RAND_KEY].choices(
                population=self._sample_coords[route_number][0],
                weights=self._sample_coords[route_number][1],
                k=length
            )

            # Build plan
            plan = []
            for coord in coords:
                order = Order()
                order.id = next(GLOBAL_ORDER_COUNTER)
                order.coord = coord
                if random[PLAN_RAND_KEY].uniform(0.0, 1.0) < self._pickup_ratio:  # Pickup order
                    # TODO: sample open_time and close_time
                    order.is_delivery = False
                else:  # Delivery order
                    # TODO: sample open_time and close_time
                    order.is_delivery = True
                plan.append(PlanElement(order=order, est_arr_time=-1, act_arr_time=-1))
            plan.append(PlanElement(rtb_fake_order, est_arr_time=-1, act_arr_time=-1))

            ret[route_number] = plan

        return ret


if __name__ == "__main__":
    loader = SamplePlanLoader(sample_config_path="C:/Users/huoranli/Downloads/fedex/1129_coord_pool")
    loader.generate_plan()
