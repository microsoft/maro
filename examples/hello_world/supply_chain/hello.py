# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ConsumerAction


def main():
    start_tick = 0
    durations = 100
    env = Env(scenario="supply_chain", topology="sample1", start_tick=start_tick, durations=durations, max_snapshots=100)
    total_episodes = 10

    matrics = None
    is_done = False

    print(env.agent_idx_list)

    for ep in range(total_episodes):
        print("Current episode:", ep)

        while not is_done:
            # This scenario require a dictionary of actions, which the key is the unit id, value if the action.
            matrics, _, is_done = env.step(None)

        env.reset()


if __name__ == "__main__":
    main()
