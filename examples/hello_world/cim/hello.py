# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action

start_tick = 0
durations = 100  # 100 days

# Initialize an environment with a specific scenario, related topology.
env = Env(scenario="cim", topology="toy.5p_ssddd_l0.0",
          start_tick=start_tick, durations=durations)


# Query environment summary, which includes business instances, intra-instance attributes, etc.
print(env.summary)

for ep in range(2):
    # Gym-like step function
    metrics, decision_event, is_done = env.step(None)

    while not is_done:
        past_week_ticks = [x for x in range(
            decision_event.tick - 7, decision_event.tick)]
        decision_port_idx = decision_event.port_idx
        intr_port_infos = ["booking", "empty", "shortage"]

        # Query the decision port booking, empty container inventory, shortage information in the past week
        past_week_info = env.snapshot_list["ports"][past_week_ticks:
                                                    decision_port_idx:
                                                    intr_port_infos]

        dummy_action = Action(decision_event.vessel_idx,
                              decision_event.port_idx, 0)

        # Drive environment with dummy action (no repositioning)
        metrics, decision_event, is_done = env.step(dummy_action)

    # Query environment business metrics at the end of an episode,
    # it is your optimized object (usually includes multi-target).
    print(f"ep: {ep}, environment metrics: {env.metrics}")
    env.reset()
