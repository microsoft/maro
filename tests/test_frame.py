# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import unittest
import numpy as np
from maro.backends.frame import node, NodeBase, NodeAttribute, FrameNode, FrameBase

from maro.utils.exception.backends_exception import (
    BackendsGetItemInvalidException,
    BackendsSetItemInvalidException,
    BackendsArrayAttributeAccessException)

from tests.utils import backends_to_test

STATIC_NODE_NUM = 5
DYNAMIC_NODE_NUM = 10

@node("static")
class StaticNode(NodeBase):
    a1 = NodeAttribute("i", 2)
    a2 = NodeAttribute("i2")
    a3 = NodeAttribute("i8")

@node("dynamic")
class DynamicNode(NodeBase):
    b1 = NodeAttribute("f")
    b2 = NodeAttribute("d")

def build_frame(enable_snapshot:bool=False, total_snapshot:int=10, backend_name="np"):

    class MyFrame(FrameBase):
        static_nodes = FrameNode(StaticNode, STATIC_NODE_NUM)
        dynamic_nodes = FrameNode(DynamicNode, DYNAMIC_NODE_NUM)

        def __init__(self):
            super().__init__(enable_snapshot=enable_snapshot, total_snapshot=total_snapshot, backend_name=backend_name)

    return MyFrame()



class TestFrame(unittest.TestCase):
    def test_node_number(self):

        """Test if node number same as defined"""
        for backend_name in backends_to_test:
            frame = build_frame(backend_name=backend_name)

            self.assertEqual(STATIC_NODE_NUM, len(frame.static_nodes), backend_name)
            self.assertEqual(DYNAMIC_NODE_NUM, len(frame.dynamic_nodes), backend_name)
    
    def test_node_accessing(self):
        """Test node accessing correct"""
        for backend_name in backends_to_test:

            frame = build_frame(backend_name=backend_name)

            # accessing for 1st node for both static and dynamic node
            static_node: StaticNode = frame.static_nodes[0]
            dynamic_node: DynamicNode = frame.dynamic_nodes[0]

            static_node.a2 = 10
            dynamic_node.b1 = 12.34

            self.assertEqual(10, static_node.a2, msg="a2 attribute should be 10 for 1st static node")
            self.assertAlmostEqual(12.34, dynamic_node.b1, 2, msg="b1 attribute should be 12.34 for 1st dynamic node")

            # check if values correct for multiple nodes
            for node in frame.static_nodes:
                node.a2 = node.index

            # check if the value correct
            for node in frame.static_nodes:
                self.assertEqual(node.index, node.a2, msg=f"static node.a2 should be {node.index}")

            # check slice accessing
            static_node.a1[1] = 12
            static_node.a1[0] = 20

            self.assertListEqual([20, 12], list(static_node.a1[:]), msg="static node's a1 should be [20, 12]")
            self.assertEqual(20, static_node.a1[0], msg="1st slot of a1 should be 20")
            self.assertEqual(12, static_node.a1[1], msg="2nd slot of a1 should be 12")

            # set again with another way
            static_node.a1[(1, 0)] = (22, 11)

            self.assertListEqual([11, 22], list(static_node.a1[:]), msg="static node a1 should be [11, 22]")

            # another way
            # NOTE: additional value will be ignored
            static_node.a1[:] = (1, 2, 3)

            self.assertListEqual([1, 2], list(static_node.a1[:]), msg="static node a1 should be [1, 2")

    def test_invalid_node_accessing(self):
        for backend_name in backends_to_test:
            frm = build_frame(backend_name=backend_name)

            static_node: StaticNode = frm.static_nodes[0]

            # get attribute value with not supported parameter
            with self.assertRaises(BackendsGetItemInvalidException) as ctx:
                a = static_node.a1["a"]

            with self.assertRaises(BackendsSetItemInvalidException) as ctx:
                static_node.a1["a"] = 1

            with self.assertRaises(BackendsArrayAttributeAccessException) as ctx:
                static_node.a1 = 1

    def test_get_node_info(self):
        for backend_name in backends_to_test:
            """Test if node information correct"""
            frm = build_frame(backend_name=backend_name)

            node_info = frm.get_node_info()

            # if should contains 2 nodes
            self.assertTrue("static" in node_info)
            self.assertTrue("dynamic" in node_info)

            # node number
            self.assertEqual(STATIC_NODE_NUM, node_info["static"]["number"])
            self.assertEqual(DYNAMIC_NODE_NUM, node_info["dynamic"]["number"])

            # check attributes
            self.assertTrue("a1" in node_info["static"]["attributes"])
            self.assertTrue("a2" in node_info["static"]["attributes"])
            self.assertTrue("a3" in node_info["static"]["attributes"])
            self.assertTrue("b1" in node_info["dynamic"]["attributes"])
            self.assertTrue("b2" in node_info["dynamic"]["attributes"])

            # check slot number
            self.assertEqual(2, node_info["static"]["attributes"]["a1"]["slots"])
            self.assertEqual(1, node_info["static"]["attributes"]["a2"]["slots"])


    def test_enable_snapshots(self):
        for backend_name in backends_to_test:
            """Test if snapshot enabled"""
            frame = build_frame(enable_snapshot=True, backend_name=backend_name)

            # snapshots should not be None
            self.assertIsNotNone(frame)

            # length should be 10
            self.assertEqual(10, len(frame.snapshots), msg="snapshot length should be 10")

            # another frame without snapshots enabled
            frame1 = build_frame(backend_name=backend_name)

            self.assertIsNone(frame1.snapshots)


    def test_reset(self):
        for backend_name in backends_to_test:
            """Test reset work as expected, reset all attributes to 0"""
            frame = build_frame(backend_name=backend_name)

            frame.static_nodes[0].a1[:] = (1, 234)

            # before reset
            self.assertListEqual([1, 234], list(frame.static_nodes[0].a1[:]), msg="static node's a1 should be [1, 234] before reset")

            frame.reset()

            # after reset
            self.assertListEqual([0, 0], list(frame.static_nodes[0].a1[:]), msg="static node's a1 should be [0, 0] after reset")

    def test_append_nodes(self):
        # NOTE: this case only support raw backend
        frame = build_frame(enable_snapshot=True, total_snapshot=10, backend_name="raw")

        # set value for last static node
        last_static_node = frame.static_nodes[-1]

        self.assertEqual(STATIC_NODE_NUM, len(frame.static_nodes))

        last_static_node.a2 = 2
        last_static_node.a3 = 9

        # this snapshot should keep 5 static nodes
        frame.take_snapshot(0)

        # append 2 new node
        frame.append_node("static", 2)

        # then there should be 2 new node instance
        self.assertEqual(STATIC_NODE_NUM + 2, len(frame.static_nodes))

        # then index should keep sequentially
        for i in range(len(frame.static_nodes)):
            self.assertEqual(i, frame.static_nodes[i].index)

        # value should be zero
        for node in frame.static_nodes[-2:]:
            self.assertEqual(0, node.a3)
            self.assertEqual(0, node.a2)
            self.assertEqual(0, node.a1[0])
            self.assertEqual(0, node.a1[1])

        last_static_node.a3 = 12

        # this snapshot should contains 7 static node
        frame.take_snapshot(1)

        static_snapshot = frame.snapshots["static"]

        # snapshot only provide current number (include delete ones)
        self.assertEqual(7, len(static_snapshot))

        # query for 1st tick
        states = static_snapshot[0::"a3"]

        # the query result of raw snapshotlist has 4 dim shape
        # (ticks, max nodes, attributes, max slots)
        self.assertTupleEqual(states.shape, (1, 7, 1, 1))

        states = states.flatten()

        # there should be 7 items, 5 for 5 nodes, 2 for padding as we do not provide node index to query, 
        # snapshotlist will padding to max_number fo node
        self.assertEqual(7, len(states))
        self.assertListEqual([0.0, 0.0, 0.0, 0.0, 9.0], list(states)[0:5])

        # 2 padding (NAN) in the end
        self.assertTrue(np.isnan(states[-2:]).all())

        states = static_snapshot[1::"a3"]

        self.assertTupleEqual(states.shape, (1, 7, 1, 1))

        states = states.flatten()

        self.assertEqual(7, len(states))

        # no padding value
        self.assertListEqual([0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0], list(states))

        # with specify node indices, will not padding to max node number
        states = static_snapshot[0:[0, 1, 2, 3, 4]:"a3"]

        self.assertTupleEqual(states.shape, (1, 5, 1, 1))

        self.assertListEqual([0.0, 0.0, 0.0, 0.0, 9.0], list(states.flatten()))

    def test_delete_node(self):
        frame = build_frame(enable_snapshot=True, total_snapshot=10, backend_name="raw")

        # set value for last static node
        last_static_node = frame.static_nodes[-1]
        second_static_node = frame.static_nodes[1]

        self.assertEqual(STATIC_NODE_NUM, len(frame.static_nodes))

        second_static_node.a3 = 444
        last_static_node.a2 = 2
        last_static_node.a3 = 9

        # this snapshot should keep 5 static nodes
        frame.take_snapshot(0)

        # delete 2nd node
        frame.delete_node(second_static_node)

        last_static_node.a3 = 123

        frame.take_snapshot(1)

        # deleted node's instance will not be removed, just mark as deleted
        self.assertTrue(second_static_node.is_deleted)

        # future setter will cause exception
        with self.assertRaises(Exception) as ctx:
            second_static_node.a3 = 11

        # attribute getter failed too
        with self.assertRaises(Exception) as ctx:
            a = second_static_node.a3

        static_snapshots = frame.snapshots["static"]

        # snapshot will try to padding to max node number if not specify node indices
        states = static_snapshots[0::"a3"]

        self.assertTupleEqual(states.shape, (1, 5, 1, 1))

        states = states.flatten()

        # no nan for 1st snapshot
        self.assertFalse(np.isnan(states).all())
        self.assertListEqual([0.0, 444.0, 0.0, 0.0, 9.0], list(states))

        states = static_snapshots[1::"a3"]

        self.assertTupleEqual(states.shape, (1, 5, 1, 1))

        states = states.flatten()

        # 2nd is padding value
        self.assertTrue(np.isnan(states[1]))

        self.assertListEqual([0.0, 0.0, 0.0, 123.0], list(states[[0, 2, 3, 4]]))

        # then we resume the deleted node, this mark it as not deleted, but values will be reset to 0
        frame.resume_node(second_static_node)
        
        # DELETE node's value will be reset after deleted
        self.assertEqual(0, second_static_node.a3)

        second_static_node.a3 = 222

        frame.take_snapshot(2)

        states = static_snapshots[2::"a3"]

        self.assertTupleEqual(states.shape, (1, 5, 1, 1))

        states = states.flatten()

        self.assertListEqual([0.0, 222.0, 0.0, 0.0, 123.0], list(states))

    def test_set_attribute_slots(self):
        frame = build_frame(enable_snapshot=True, total_snapshot=10, backend_name="raw")

        # set value for last static node
        last_static_node = frame.static_nodes[-1]

        last_static_node.a2 = 2
        last_static_node.a3 = 9
        last_static_node.a1[:] = (0, 1)

        frame.take_snapshot(0)

        # change slots number

        # we do not accept 0 slots
        with self.assertRaises(Exception) as ctx:
            frame.set_attribute_slot("static", "a1", 0)

        # do not support set for not exist node
        with self.assertRaises(Exception) as ctx:
            frame.set_attribute_slot("no", "a1", 12)

        # do not support set for not exist attribute
        with self.assertRaises(Exception) as ctx:
            frame.set_attribute_slot("static", "aa", 1)

        # extend slots
        frame.set_attribute_slot("static", "a2", 4)

        last_static_node.a2[3] = 0

if __name__ == "__main__":
    unittest.main()
