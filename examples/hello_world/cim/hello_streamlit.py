# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from random import randint, seed

from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action, ActionType

if __name__ == "__main__":
    seed(0)
    NUM_EPISODE = 2

    opts = dict()
    """
    enable-dump-snapshot parameter means business_engine needs dump snapshot data before reset.
    If you leave value to empty string, it will dump to current folder.
    For getting dump data, please uncomment below line and specify dump destination folder.
    """
    opts["enable-dump-snapshot"] = "YOUR_FOLDER_NAME"

    # Initialize an environment with a specific scenario, related topology.
    env = Env(
        scenario="cim",
        topology="global_trade.22p_l0.1",
        start_tick=0,
        durations=100,
        options=opts,
    )

    # To reset environmental data before starting a new experiment.
    env.reset()

    for ep in range(NUM_EPISODE):
        # Gym-like step function.
        metrics, decision_event, is_done = env.step(None)

        while not is_done:
            action_scope = decision_event.action_scope
            to_discharge = action_scope.discharge > 0 and randint(0, 1) > 0

            random_action = Action(
                decision_event.vessel_idx,
                decision_event.port_idx,
                randint(0, action_scope.discharge if to_discharge else action_scope.load),
                ActionType.DISCHARGE if to_discharge else ActionType.LOAD,
            )

            # Drive environment with dummy action (no repositioning)
            metrics, decision_event, is_done = env.step(random_action)

        # Query environment business metrics at the end of an episode,
        # it is your optimized object (usually includes multi-target).
        print(f"ep: {ep}, environment metrics: {env.metrics}")

        env.reset()
