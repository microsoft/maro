# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator import Env
from maro.simulator.scenarios.oncall_routing.common import OncallRoutingPayload


if __name__ == "__main__":
    env = Env(
        scenario="oncall_routing", topology="example", start_tick=0, durations=1440,
    )

    env.reset(keep_seed=True)
    metrics, decision_event, is_done = env.step(None)
    while not is_done:
        assert isinstance(decision_event, OncallRoutingPayload)
        orders = decision_event.oncall_orders
        print(
            f"Env Tick: {env.tick}, "
            f"On-call Order Num: {len(orders)}, "
            f"{(orders[0].id, orders[0].coord) if len(orders) > 0 else ''}"
        )
        metrics, decision_event, is_done = env.step(None)

    print(metrics)
