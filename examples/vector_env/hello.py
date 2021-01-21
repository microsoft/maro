# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.vector_env.vector_env import VectorEnv


if __name__ == "__main__":
    with VectorEnv(2, scenario="cim", topology="toy.5p_ssddd_l0.0", durations=100) as env:
        for ep in range(1):
            print("current episode:", ep)

            metrics, decision_event, is_done = (None, None, False)

            while not is_done:
                metrics, decision_event, is_done = env.step(None)

                print("metrics:", metrics)
                print("decision events:", decision_event)
                print("done:", is_done)

            empty_list = env.snapshot_list["ports"][::"empty"]

            print(empty_list)

            env.reset()
