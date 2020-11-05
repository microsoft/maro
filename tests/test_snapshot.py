# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import unittest

from test_frame import DYNAMIC_NODE_NUM, STATIC_NODE_NUM, build_frame


class TestFrame(unittest.TestCase):
    def test_take_snapshot(self):
        """Test if take_stapshot work"""
        frame = build_frame(True)

        # 1st static node
        static_node = frame.static_nodes[0]

        static_node.a1[:] = [1, 23]

        frame.take_snapshot(0)

        a1_at_tick_0 = frame.snapshots["static"][:0:"a1"]

        # the value should be same with current 
        self.assertListEqual(list(a1_at_tick_0.astype("i")), [1, 23], msg="1st static node's a1 should be [1, 23] at tick 0")

        # test if the value in snapshot will be changed after change frame
        static_node.a1[1] = 123

        a1_at_tick_0 = frame.snapshots["static"][:0:"a1"]

        self.assertListEqual(list(a1_at_tick_0.astype("i")), [1, 23], msg="1st static node's a1 should be [1, 23] at tick 0 even static node value changed")

    def test_slice_quering(self):
        """Test if states quering result correct"""
        frame = build_frame(True, total_snapshot=2)

        # one node changes
        static_node = frame.static_nodes[0]

        static_node.a2 = 1

        # before takeing snapshot, states should be 0
        static_node_a2_states = frame.snapshots["static"][0:0:"a2"]

        self.assertEqual(1, len(static_node_a2_states), msg="slicing with 1 tick, 1 node and 1 attr, should return array with 1 result")
        self.assertEqual(0, static_node_a2_states.astype("i")[0], msg="states before taking snashot should be 0")

        frame.take_snapshot(0)

        # set a2 and a3 for all static nodes
        for i, node in enumerate(frame.static_nodes):
            node.a3 = 100 * i
            node.a2 = 100 * i + 1

        # take snapshot
        frame.take_snapshot(1)

        # query with 2 attributes
        states = frame.snapshots["static"][1::["a3", "a2"]]

        # with this quering, the result should be like
        # so we can reshape it as
        states = states.reshape(len(frame.static_nodes), 2)

        # then 1st should a3 value for all static node
        # 2nd should be a2 value for all static node
        self.assertListEqual(list(states[:, 0].astype("i")), [100 * i for i in range(len(frame.static_nodes))], msg="1st row of states should be a3 value")
        self.assertListEqual(list(states[:, 1].astype("i")), [100 * i + 1 for i in range(len(frame.static_nodes))], msg="2nd row of states should be a2 value")

        # quering without tick, means return all ticks in snapshot
        states = frame.snapshots["static"][:0:"a2"]

        # reshape it as 2-dim, so row is tick
        states = states.reshape(2, -1).astype("i")

        # then each row is a2 value for 1st static node at that tick
        self.assertEqual(1, len(states[0]), msg="1st static should contains 1 a2 value at tick 0")

        self.assertEqual(1, states[0], msg="1st static node a2 value should be 1 at tick 0")

        self.assertEqual(1, states[1], msg="1st staic node a2 value should be 1 at tick 1")

        # quering without node index, means return attributes of all the nodes
        states = frame.snapshots["static"][1::"a2"]

        self.assertEqual(len(frame.static_nodes), len(states), msg="1 tick 1 attribute and not specified ticks, should return array length same as node number")
        self.assertListEqual(list(states.astype("i")), [100 * i + 1 for i in range(len(frame.static_nodes))], msg="a2 at 1st row should be values at tick 1")

        # when reach the max size of snapshot, oldest one will be overwrite
        static_node.a2 = 1000

        frame.take_snapshot(2)

        # check if current snapshot max size correct
        self.assertEqual(2, len(frame.snapshots), msg="snapshot list max size should be 2")

        # and result without ticks should return 2 row: 2*len(static_nodes)
        states = frame.snapshots["static"][::"a2"]
        states = states.reshape(-1, len(frame.static_nodes))

        self.assertEqual(2, len(states), msg="states should contains 2 row")

        # 1st row should be values at tick 1
        self.assertListEqual(list(states[0].astype("i")), [100 * i + 1 for i in range(len(frame.static_nodes))], msg="a2 at tick 1 for all nodes should be correct")
        
        # 2nd row should be lastest one
        self.assertEqual(1000, states[1][0], msg="a2 for 1st static node for 2nd row should be 1000")

        # quering with ticks that being over-wrote, should return 0 for that tick
        states = frame.snapshots["static"][(0, 1, 2)::"a2"]
        states = states.reshape(-1, len(frame.static_nodes))

        self.assertEqual(3, len(states), msg="states should contains 3 row")
        self.assertListEqual([0]*len(frame.static_nodes), list(states[0].astype("i")), msg="over-wrote tick should return 0")
        self.assertListEqual(list(states[1].astype("i")), [100 * i + 1 for i in range(len(frame.static_nodes))], msg="a2 at tick 1 for all nodes should be correct")
        self.assertEqual(1000, states[2][0], msg="a2 for 1st static node for 2nd row should be 1000")  

    def test_snapshot_length(self):
        """Test __len__ function result"""

        frm = build_frame(True, total_snapshot=10)

        self.assertEqual(10, len(frm.snapshots))


    def test_snapshot_node_length(self):
        """Test if node number in snapshot correct"""

        frm = build_frame(True)

        self.assertEqual(STATIC_NODE_NUM, len(frm.snapshots["static"]))
        self.assertEqual(DYNAMIC_NODE_NUM, len(frm.snapshots["dynamic"]))

    def test_quering_with_not_exist_indices(self):
        # NOTE: when quering with not exist indices, snapshot will try to fill the result of that index with 0

        frm = build_frame(True)

        for node in frm.static_nodes:
            node.a2 = node.index

        frm.take_snapshot(0)

        # with 1 invalid index, all should be 0
        states = frm.snapshots["static"][1::"a2"]

        self.assertListEqual(list(states.astype("I")), [0]*STATIC_NODE_NUM)

        # with 1 invalid index, one valid index
        states = frm.snapshots["static"][(0, 1)::"a2"]
        states = states.reshape(-1, STATIC_NODE_NUM)

        # index 0 should be same with out current value
        self.assertListEqual(list(states[0].astype("i")), [i for i in range(STATIC_NODE_NUM)])
        self.assertListEqual(list(states[1].astype("i")), [0]*STATIC_NODE_NUM)

    def test_get_attribute_with_undefined_attribute(self):
        frm = build_frame(True)
        frm.take_snapshot(0)

        # not exist attribute name
        with self.assertRaises(ValueError) as ctx:
            states = frm.snapshots["static"][::"a8"]

        # not exist node name
        self.assertIsNone(frm.snapshots["hehe"])


if __name__ == "__main__":
    unittest.main()
