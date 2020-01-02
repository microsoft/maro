# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.simulator import Env
from maro.simulator.scenarios.ecr.common import Action

env = Env(scenario='ecr', topology='5p_ssddd_l0.0', max_tick=10)

for ep in range(2):
    _, decision_event, is_done = env.step(None)

    while not is_done:
        print(f'ep: {ep}, decision event: {decision_event}')
        print(f"shortage of port {decision_event.port_idx}: {env.snapshot_list.static_nodes[decision_event.tick: decision_event.port_idx: ('shortage', 0)]}")
        dummy_action = Action(decision_event.vessel_idx, decision_event.port_idx, 0)
        reward, decision_event, is_done = env.step(dummy_action)

    env.reset()