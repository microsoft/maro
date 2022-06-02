# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from time import time

from termgraph import termgraph as tg

from maro.backends.frame import FrameBase, FrameNode, NodeAttribute, NodeBase, node

NODE1_NUMBER = 100
NODE2_NUMBER = 100
MAX_TICK = 10000


READ_WRITE_NUMBER = 10000000
STATES_QURING_TIME = 10000
TAKE_SNAPSHOT_TIME = 10000

AVG_TIME = 4


@node("node1")
class TestNode1(NodeBase):
    a = NodeAttribute("i")
    b = NodeAttribute("i")
    c = NodeAttribute("i")
    d = NodeAttribute("i")
    e = NodeAttribute("i", 16)


@node("node2")
class TestNode2(NodeBase):
    b = NodeAttribute("i", 20)


class TestFrame(FrameBase):
    node1 = FrameNode(TestNode1, NODE1_NUMBER)
    node2 = FrameNode(TestNode2, NODE2_NUMBER)

    def __init__(self, backend_name):
        super().__init__(
            enable_snapshot=True,
            total_snapshot=TAKE_SNAPSHOT_TIME,
            backend_name=backend_name,
        )


def build_frame(backend_name: str):
    return TestFrame(backend_name)


def attribute_access(frame, times: int):
    """Return time cost (in seconds) for attribute acceesing test"""
    start_time = time()

    n1 = frame.node1[0]

    for _ in range(times):
        n1.a
        n1.a = 12

    return time() - start_time


def take_snapshot(frame, times: int):
    """Return times cost (in seconds) for take_snapshot operation"""

    start_time = time()

    for i in range(times):
        frame.take_snapshot(i)

    return time() - start_time


def snapshot_query(frame, times: int):
    """Return time cost (in seconds) for snapshot querying"""

    start_time = time()

    for i in range(times):
        states = frame.snapshots["node1"][i::"a"]

    return time() - start_time


if __name__ == "__main__":
    chart_colors = [91, 94]

    chart_args = {
        "filename": "-",
        "title": "Performance comparison between cpp and np backends",
        "width": 40,
        "format": "{:<5.2f}",
        "suffix": "",
        "no_labels": False,
        "color": None,
        "vertical": False,
        "stacked": False,
        "different_scale": False,
        "calendar": False,
        "start_dt": None,
        "custom_tick": "",
        "delim": "",
        "verbose": False,
        "version": False,
        "histogram": False,
        "no_values": False,
    }

    chart_labels = [
        f"attribute accessing ({READ_WRITE_NUMBER})",
        f"take snapshot ({STATES_QURING_TIME})",
        f"states querying ({STATES_QURING_TIME})",
    ]

    chart_data = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    i = 0
    j = 0

    for backend_name in ["static", "dynamic"]:
        frame = build_frame(backend_name)

        j = 0

        for func, args in [
            (attribute_access, READ_WRITE_NUMBER),
            (take_snapshot, TAKE_SNAPSHOT_TIME),
            (snapshot_query, STATES_QURING_TIME),
        ]:
            t = func(frame, args)

            chart_data[j][i] = t

            j += 1

        i += 1

    tg.print_categories(["static", "dynamic"], chart_colors)
    tg.chart(chart_colors, chart_data, chart_args, chart_labels)
