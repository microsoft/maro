# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from time import time

from maro.simulator import Env
from maro.simulator.scenarios.cim.frame_builder import gen_cim_frame

"""
In this file we will test performance for frame, snapshotlist, and cim scenario, with following config

1. ports: 100
2. vessels: 100
3. max_tick: 10000

"""

PORTS_NUMBER = 100
VESSELS_NUMBER = 100
MAX_TICK = 10000
STOP_NUMBER = (6, 6)

READ_WRITE_NUMBER = 1000000
STATES_QURING_TIME = 100000

def test_frame_only():
    start_time = time()

    frm = gen_cim_frame(PORTS_NUMBER, VESSELS_NUMBER, STOP_NUMBER, MAX_TICK)

    static_node = frm.ports[0]

    # read & write one attribute N times with simplified interface
    for _ in range(READ_WRITE_NUMBER):
        static_node.a2 = 10
        a = static_node.a2

    end_time = time()

    print(f"node read & write {READ_WRITE_NUMBER} times: {end_time - start_time}")


def test_snapshot_list_only():
    frm = gen_cim_frame(PORTS_NUMBER, VESSELS_NUMBER, STOP_NUMBER, MAX_TICK)

    start_time = time()

    # 1. take snapshot
    for i in range(MAX_TICK):
        frm.take_snapshot(i)

    end_time = time()

    print(f"take {MAX_TICK} snapshot: {end_time - start_time}")


def test_states_quering():
    frm = gen_cim_frame(PORTS_NUMBER, VESSELS_NUMBER, STOP_NUMBER, MAX_TICK)
    frm.take_snapshot(0)

    start_time = time()

    static_ss = frm.snapshots["ports"]

    for i in range(STATES_QURING_TIME):
        states = static_ss[::"empty"]

    end_time = time()

    print(f"Single state quering {STATES_QURING_TIME} times: {end_time - start_time}")


def test_cim():
    eps = 4

    env = Env("cim", "toy.5p_ssddd_l0.0", durations=MAX_TICK)

    start_time = time()

    for _ in range(eps):
        _, _, is_done = env.step(None)

        while not is_done:
            _, _, is_done = env.step(None)

        env.reset()

    end_time = time()

    print(f"cim 5p toplogy with {MAX_TICK} total time cost: {(end_time - start_time)/eps}")


if __name__ == "__main__":
    test_frame_only()
    test_snapshot_list_only()
    test_states_quering()
    test_cim()
