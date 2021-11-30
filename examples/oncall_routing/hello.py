# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator import Env
from maro.simulator.scenarios.oncall_routing.common import Action, OncallRoutingPayload


if __name__ == "__main__":
    env = Env(
        scenario="oncall_routing", topology="example", start_tick=0, durations=1000,
        # options={"config_path": "C:/workspace/fedex_topology/example_sample/"}
    )

    env.reset()
    metrics, decision_event, is_done = env.step(None)
    while not is_done:
        assert isinstance(decision_event, OncallRoutingPayload)
        orders = decision_event.oncall_orders
        if len(orders) > 0:
            print(env.tick)

            for order in orders:
                print(order.id, order.coord)

            break
