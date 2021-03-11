import os, psutil

import sys
import numpy as np
import tcod
import pprint

from tabulate import tabulate
from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ConsumerAction
from timeit import timeit


def go(env: Env):
    env.reset()

    is_done = False

    while not is_done:
        _, _, is_done = env.step(None)


if __name__ == "__main__":
    topology = sys.argv[1]
    durations = int(sys.argv[2] if len(sys.argv) > 2 else 100)

    env = Env(scenario="supply_chain", topology=topology, durations=durations)

    print(f"config: f{topology}, steps: {durations}")

    print("avg time cost: ", timeit(lambda : go(env), number=2))

    process = psutil.Process(os.getpid())

    memory = process.memory_info().rss/1024/1024
    
    print("memory cost: ", memory)