# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action

start_tick = 0
durations = 100  # 100 days

opts = dict()

# Initialize an environment with a specific scenario, related topology.
env = Env(scenario="supply_chain", topology="toy.5p_ssddd_l0.0",
          start_tick=start_tick, durations=durations, options=opts)

for ep in range(1):
    metrics, decision_event, is_done = (None, None, False)

    while not is_done:
        metrics, decision_event, is_done = env.step(None)

    env.reset()
