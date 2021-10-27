# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator import Env
from maro.simulator.scenarios.hvac.common import Action, PendingDecisionPayload


if __name__ == "__main__":
    # env = Env(scenario="hvac", topology="building121", start_tick=0, durations=4630)
    env = Env(scenario="hvac", topology="building121", start_tick=0, durations=10)
    env.reset()
    # print(env.summary)

    attributes = ["mat", "dat", "at", "kw", "sps", "das"]

    for ep in range(1):
        metrics, pending_decision_payload, is_done = env.step(None)
        info = env.snapshot_list["ahus"][::attributes].reshape(env.tick + 1, -1)
        print(env.tick, info)

        while not is_done:
            dummy_action = Action(
                ahu_idx=pending_decision_payload.ahu_idx,
                sps=10,
                das=10
            )

            metrics, pending_decision_payload, is_done = env.step(dummy_action)

        print(f"Ep: {ep}, env metrics: {metrics}")
        info = env.snapshot_list["ahus"][::attributes].reshape(env.tick + 1, -1)
        print(attributes)
        print(info)

        env.reset()
