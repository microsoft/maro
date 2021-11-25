# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator import Env
from maro.simulator.scenarios.oncall_routing.common import Action, OncallRoutingPayload


if __name__ == "__main__":
    env = Env(scenario="oncall_routing", topology="example", start_tick=0, durations=10000000)

    env.reset()
    metrics, decision_event, is_done = env.step(None)
    while not is_done:
        assert isinstance(decision_event, OncallRoutingPayload)
        orders = decision_event.oncall_orders
        if len(orders) > 0:
            print(env.tick)

            _actions = [
                Action(order_id=orders[0].id, route_number=987, insert_index=3),
                Action(order_id=orders[1].id, route_number=987, insert_index=5)
            ]

            env.step(_actions)
            break
