# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import sys
from maro.simulator import Env


def main(topology_name: str):
    durations = 100

    env = Env(scenario="supply_chain", topology=topology_name, durations=durations, max_snapshots=100)

    total_episodes = 10

    is_done = False

    for ep in range(total_episodes):
        print("Current episode:", ep)

        while not is_done:
            # This scenario require a dictionary of actions, which the key is the unit id, value if the action.
            _, _, is_done = env.step(None)

        env.reset()

        is_done = False


if __name__ == "__main__":
    topology_name = "sample"

    if len(sys.argv) > 1:
        topology_name = sys.argv[1]

    print("running topology:", topology_name)

    main(topology_name)
