# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import unittest

from maro.simulator.graph import Graph, SnapshotList, GraphAttributeType, ResourceNodeType, GraphAttributeNotFoundError, \
    SnapshotSliceError

STATIC_NODE_NUM = 10
DYNAMIC_NODE_NUM = 10
MAX_TICK = 10


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(STATIC_NODE_NUM, DYNAMIC_NODE_NUM)

        self.graph.register_attribute("a1", GraphAttributeType.FLOAT, 1)
        self.graph.register_attribute("a2", GraphAttributeType.INT, 1)
        self.graph.register_attribute("m1", GraphAttributeType.INT_MAT, 10, 2, 5)

        self.graph.setup()

        # init value
        self.graph.set_attribute(ResourceNodeType.STATIC, 0, "a1", 0, 123)
        self.graph.set_attribute(ResourceNodeType.DYNAMIC, 0, "a1", 0, 1234)

        self.snapshots = SnapshotList(self.graph, MAX_TICK)

    def test_get_attribute_correct(self):
        # take a snapshot first
        self.snapshots.insert_snapshot(self.graph, 0)

        v = self.snapshots.get_attribute(0, ResourceNodeType.STATIC, 0, "a1", 0)

        self.assertEqual(v, 123)

        v = self.snapshots.get_attribute(0, ResourceNodeType.DYNAMIC, 0, "a1", 0)

        self.assertEqual(v, 1234)

    def test_take_snapshot_correct(self):
        self.snapshots.insert_snapshot(self.graph, 0)  # take snapshot for tick 0

        v = self.snapshots.get_attribute(0, ResourceNodeType.STATIC, 0, "a1", 0)

        self.assertEqual(v, 123)

        # test if the value in snapshot will be changed after change graph
        self.graph.set_attribute(ResourceNodeType.STATIC, 0, "a1", 0, 1234)

        v = self.graph.get_attribute(ResourceNodeType.STATIC, 0, "a1", 0)

        self.assertEqual(v, 1234)

        v = self.snapshots.get_attribute(1, ResourceNodeType.STATIC, 0, "a1", 0)

        self.assertEqual(v, 123)

    def test_get_attributes_correct(self):
        self.snapshots.insert_snapshot(self.graph, 0)

        # change the value and insert another snapshot
        self.graph.set_attribute(ResourceNodeType.STATIC, 0, "a1", 0, 1234)

        self.snapshots.insert_snapshot(self.graph, 1)

        # query 1st slot of a1 at tick (0, 1)
        states = self.snapshots.get_attributes(ResourceNodeType.STATIC, [0, 1], [0, ], ["a1", ], [0, ])

        self.assertEqual(states[0], 123)  # value at tick 0
        self.assertEqual(states[1], 1234)  # value at tick 1

    def test_get_attributes_with_invalid_ticks(self):
        self.snapshots.insert_snapshot(self.graph, 0)

        states = self.snapshots.get_attributes(ResourceNodeType.STATIC, [-1, 0, 1], [0, ], ["a1", ], [0, ])

        # since we will padding the value for invalid ticks, so it should return numpy array that shape is (3,)
        self.assertEqual(states.shape, (3,))

        # and the value at invalid tick should be 0
        self.assertEqual(states[0], 0)
        self.assertEqual(states[1], 123)
        self.assertEqual(states[2], 0)

    def test_get_attribute_with_undefined_attribute(self):
        self.snapshots.insert_snapshot(self.graph, 0)

        # get_attribute method require that the attribute must be registered
        with self.assertRaises(GraphAttributeNotFoundError) as ctx:
            self.snapshots.get_attribute(1, ResourceNodeType.STATIC, 0, "a3", 0)

    def test_attribute_info(self):
        attrs = self.snapshots.attributes

        self.assertEqual("a1", attrs[0]["name"])
        self.assertEqual(1, attrs[0]["slot length"])

        self.assertEqual("a2", attrs[1]["name"])
        self.assertEqual(1, attrs[1]["slot length"])

        self.assertEqual(1, self.snapshots.get_slot_length("a1"))

    def test_get_matrix(self):
        self.snapshots.insert_snapshot(self.graph, 0)

        for i in range(2):
            for j in range(5):
                self.graph.set_int_matrix_value("m1",  i, j, i*10 + j)

        self.snapshots.insert_snapshot(self.graph, 1)

        # check matrix value at tick 1
        v = self.snapshots.matrix[1: "m1"]

        for i in range(2):
            for j in range(5):
                self.assertEqual(i*10+j, v[0][i*5+j])

        # check matrix value at tick 0
        v = self.snapshots.matrix[0: "m1"]

        for i in range(2):
            for j in range(5):
                self.assertEqual(0, v[0][i*5+j])

    def test_snapshot_slice(self):
        self.snapshots.insert_snapshot(self.graph, 0)

        # change the value and insert another snapshot
        self.graph.set_attribute(ResourceNodeType.STATIC, 0, "a1", 0, 1234)

        self.snapshots.insert_snapshot(self.graph, 1)

        ticks = [0, 1]
        ids = [0]
        attrs = ["a1"]
        slots = [0]

        # query 1st slot of a1 at tick (0, 1)

        states = self.snapshots.get_attributes(ResourceNodeType.STATIC, ticks, ids, attrs, slots)

        v = self.snapshots.static_nodes[ticks: ids: (attrs, slots)]
        v = self.snapshots.static_nodes[ticks: 0: ("a1", 0)]
        v = self.snapshots.static_nodes[: 0: ("a1", 0)]
        v = self.snapshots.static_nodes[0:: ("a1", 0)]
        v = self.snapshots.static_nodes[::("a1", 0)]

        self.assertEqual(states[0], 123)  # value at tick 0
        self.assertEqual(states[1], 1234)  # value at tick 1

        v = self.snapshots.static_nodes[ticks: ids: (attrs, slots)]

        self.assertEqual(states[0], v[0])
        self.assertEqual(states[1], v[1])

        # test with short version
        v = self.snapshots.static_nodes[ticks: 0: ("a1", 0)]

        self.assertEqual(states[0], v[0])
        self.assertEqual(states[1], v[1])

        # more on short version
        v = self.snapshots.static_nodes[ticks: 0: ("a1", [0])]

        self.assertEqual(states[0], v[0])
        self.assertEqual(states[1], v[1])

        # more on short version
        # we should provide attributes names and indices, cannot get all
        with self.assertRaises(SnapshotSliceError) as ctx:
            v = self.snapshots.static_nodes[::]

        # get all the ticks
        v = self.snapshots.static_nodes[: 0: ("a1", 0)]

        self.assertEqual(states[0], v[0])
        self.assertEqual(states[1], v[1])

        # get all nodes at tick 0
        v = self.snapshots.static_nodes[0:: ("a1", 0)]

        self.assertEqual(STATIC_NODE_NUM, len(v))

        # at tick 0, only static node 0 have value 123
        self.assertEqual(123, v[0])

        for i in range(1, STATIC_NODE_NUM):
            self.assertEqual(0, v[i])

        # all all nodes at all the ticks
        v = self.snapshots.static_nodes[::("a1", 0)]

        # we have insert 2 ticks
        self.assertEqual(STATIC_NODE_NUM * 2, len(v))

        # at tick 0, only static node 0 have value 123
        self.assertEqual(123, v[0])

        # at tick 1, the value is 1234
        self.assertEqual(1234, v[STATIC_NODE_NUM])

        for i in range(len(v)):
            if i != 0 and i != STATIC_NODE_NUM:
                self.assertEqual(0, v[i])


if __name__ == "__main__":
    unittest.main()
