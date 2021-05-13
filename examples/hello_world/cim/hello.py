# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


# Enable realtime data streaming with following statements.

import os

os.environ["MARO_STREAMIT_ENABLED"] = "true"
os.environ["MARO_STREAMIT_EXPERIMENT_NAME"] = "test_423_3"


from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action, ActionType
from maro.streamit import streamit

if __name__ == "__main__":
    start_tick = 0
    durations = 3  # 100 days

    opts = dict()
    with streamit:
        """
        enable-dump-snapshot parameter means business_engine needs dump snapshot data before reset.
        If you leave value to empty string, it will dump to current folder.
        For getting dump data, please uncomment below line and specify dump destination folder.
        """
        # opts['enable-dump-snapshot'] = ''

        # Initialize an environment with a specific scenario, related topology.
        env = Env(scenario="cim", topology="global_trade.22p_l0.1",
                  start_tick=start_tick, durations=durations, options=opts)
        env.reset()
        # Query environment summary, which includes business instances, intra-instance attributes, etc.
        print(env.summary)

        for ep in range(2):
            # Tell streamit we are in a new episode.
            streamit.episode(ep)

            # Gym-like step function.
            metrics, decision_event, is_done = env.step(None)

            while not is_done:
                past_week_ticks = [x for x in range(
                    max(decision_event.tick - 7, 0), decision_event.tick)]
                decision_port_idx = decision_event.port_idx
                intr_port_infos = ["booking", "empty", "shortage"]

                # Query the decision port booking, empty container inventory, shortage information in the past week
                past_week_info = env.snapshot_list["ports"][past_week_ticks:
                                                            decision_port_idx:
                                                            intr_port_infos]

                dummy_action = Action(
                    decision_event.vessel_idx,
                    decision_event.port_idx,
                    0,
                    ActionType.LOAD
                )

                # Drive environment with dummy action (no repositioning)
                metrics, decision_event, is_done = env.step(dummy_action)

            # Query environment business metrics at the end of an episode,
            # it is your optimized object (usually includes multi-target).
            print(f"ep: {ep}, environment metrics: {env.metrics}")

            env.reset()
