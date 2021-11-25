# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List

import pandas as pd

from maro.simulator.scenarios.oncall_routing.common import PlanElement, RouteNumber
from maro.simulator.scenarios.oncall_routing.order import GLOBAL_ORDER_COUNTER, Order


rtb_fake_order = Order()
rtb_fake_order.id = next(GLOBAL_ORDER_COUNTER)
rtb_fake_order.coord = (32.72329226, -117.0718922)


def load_plan_simple(csv_path: str) -> Dict[RouteNumber, List[PlanElement]]:
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
            order.coord = (e["LAT"], e["LNG"])
            order.open_time = e["READYTIME"]
            order.close_time = e["CLOSETIME"]
            order.is_delivery = e["IS_DELIVERY"]

            plan.append(PlanElement(order=order, est_arr_time=-1, act_arr_time=-1))
        plan.append(PlanElement(rtb_fake_order, est_arr_time=-1, act_arr_time=-1))
        plan_by_route[route_number] = plan

    print(f"Loading finished. Loaded data of {len(plan_by_route)} routes.")
    return plan_by_route


if __name__ == "__main__":
    load_plan_simple("C:/workspace/maro/maro/simulator/scenarios/oncall_routing/topologies/example/routes.csv")
