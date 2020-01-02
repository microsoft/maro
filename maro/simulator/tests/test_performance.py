# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from time import time

from maro.simulator.graph import SnapshotList, ResourceNodeType

from maro.simulator.core import Env
from maro.simulator.scenarios.ecr.common import Action
from maro.simulator.scenarios.ecr.graph_builder import gen_ecr_graph

'''
in this file we will test performance for graph, snapshotlist, and ecr scenario,with following config

1. dynamic node: 100
2. static node: 100
3. max_tick: 1000

'''

DYNAMIC_NODE_NUMBER = 100
STATIC_NODE_NUMBER = 100
MAX_TICK = 1000
READ_WRITE_NUMBER = 1000000
STOP_NUMBER = (4, 4)


def test_graph_only():
    start_time = time()

    g = gen_ecr_graph(STATIC_NODE_NUMBER, DYNAMIC_NODE_NUMBER, STOP_NUMBER)

    # read & write one attribute N times with simplified interface
    for _ in range(READ_WRITE_NUMBER):
        # NOTE: this is slow attribute accessing interface, we will typed interface later
        v = g.get_attribute(ResourceNodeType.STATIC, 0, "empty", 0)
        g.set_attribute(ResourceNodeType.STATIC, 0, 'empty', 0, v + 1)

    end_time = time()

    print(f"graph total time cost: {end_time - start_time}")


def test_snapshot_list_only():
    g = gen_ecr_graph(STATIC_NODE_NUMBER, DYNAMIC_NODE_NUMBER, STOP_NUMBER)

    ss = SnapshotList(g, MAX_TICK)

    start_time = time()

    # 1. take snapshot
    for i in range(MAX_TICK):
        ss.insert_snapshot(g, i)

    step_1_end_time = time()

    print(f"take {MAX_TICK} snapshot, cost: {step_1_end_time - start_time}")

    # 2. get state that same with single model state shaping N times
    ticks = [-2, -1, 0, 1, 2]
    node_ids = [0, 1, 2]
    attrs = ["empty", "laden", "on_shipper", "on_consignee"]
    indices = [0]
    for i in range(MAX_TICK):
        state = ss.get_attributes(ResourceNodeType.STATIC, ticks, node_ids, attrs, indices)

    step_2_end_time = time()

    print(f"get {MAX_TICK} times of state with raw function call, cost: {step_2_end_time - step_1_end_time}")

    for i in range(MAX_TICK):
        state = ss.static_nodes[ticks: node_ids: (attrs, indices)]

    step_3_end_time = time()

    print(f"get {MAX_TICK} times of state with slice interface, cost: {step_3_end_time - step_2_end_time}")

    print(f"snapshot total time cost: {step_2_end_time - start_time}")


def test_ecr():
    start_time = time()

    env = Env("ecr", "5p_ssddd_l0.0", 1000)

    _, _, is_done = env.step(None)

    while not is_done:
        _, _, is_done = env.step(Action(0, 0, 0))

    end_time = time()

    print(f"env total time cost: {end_time - start_time}")


test_graph_only()
test_snapshot_list_only()
test_ecr()
