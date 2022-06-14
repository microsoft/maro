# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import unittest

import numpy as np
import pandas as pd

from maro.backends.backend import AttributeType
from maro.backends.frame import FrameBase, FrameNode, NodeAttribute, NodeBase, node
from maro.utils.exception.backends_exception import (
    BackendsArrayAttributeAccessException,
    BackendsGetItemInvalidException,
    BackendsSetItemInvalidException,
)

from tests.utils import backends_to_test

STATIC_NODE_NUM = 5
DYNAMIC_NODE_NUM = 10


@node("static")
class StaticNode(NodeBase):
    a1 = NodeAttribute("i", 2)
    a2 = NodeAttribute(AttributeType.Short)
    a3 = NodeAttribute(AttributeType.Long)


@node("dynamic")
class DynamicNode(NodeBase):
    b1 = NodeAttribute(AttributeType.Float)
    b2 = NodeAttribute(AttributeType.Double)


def build_frame(enable_snapshot: bool = False, total_snapshot: int = 10, backend_name="static"):
    class MyFrame(FrameBase):
        static_nodes = FrameNode(StaticNode, STATIC_NODE_NUM)
        dynamic_nodes = FrameNode(DynamicNode, DYNAMIC_NODE_NUM)

        def __init__(self):
            super().__init__(
                enable_snapshot=enable_snapshot,
                total_snapshot=total_snapshot,
                backend_name=backend_name,
            )

    return MyFrame()


class TestFrame(unittest.TestCase):
    def test_node_number(self):
        """Test if node number same as defined"""
        for backend_name in backends_to_test:
            frame = build_frame(backend_name=backend_name)

            self.assertEqual(
                STATIC_NODE_NUM,
                len(
                    frame.static_nodes,
                ),
                backend_name,
            )
            self.assertEqual(
                DYNAMIC_NODE_NUM,
                len(
                    frame.dynamic_nodes,
                ),
                backend_name,
            )

    def test_node_accessing(self):
        """Test node accessing correct"""
        for backend_name in backends_to_test:

            frame = build_frame(backend_name=backend_name)

            # accessing for 1st node for both static and dynamic node
            static_node: StaticNode = frame.static_nodes[0]
            dynamic_node: DynamicNode = frame.dynamic_nodes[0]

            static_node.a2 = 10
            dynamic_node.b1 = 12.34

            self.assertEqual(
                10,
                static_node.a2,
                msg="a2 attribute should be 10 for 1st static node",
            )
            self.assertAlmostEqual(
                12.34,
                dynamic_node.b1,
                2,
                msg="b1 attribute should be 12.34 for 1st dynamic node",
            )

            # check if values correct for multiple nodes
            for node in frame.static_nodes:
                node.a2 = node.index

            # check if the value correct
            for node in frame.static_nodes:
                self.assertEqual(
                    node.index,
                    node.a2,
                    msg=f"static node.a2 should be {node.index}",
                )

            # check slice accessing
            static_node.a1[1] = 12
            static_node.a1[0] = 20

            self.assertListEqual(
                [20, 12],
                list(
                    static_node.a1[:],
                ),
                msg="static node's a1 should be [20, 12]",
            )
            self.assertEqual(
                20,
                static_node.a1[0],
                msg="1st slot of a1 should be 20",
            )
            self.assertEqual(
                12,
                static_node.a1[1],
                msg="2nd slot of a1 should be 12",
            )

            # set again with another way
            static_node.a1[(1, 0)] = (22, 11)

            self.assertListEqual(
                [11, 22],
                list(
                    static_node.a1[:],
                ),
                msg="static node a1 should be [11, 22]",
            )

            # another way
            # NOTE: additional value will be ignored
            static_node.a1[:] = (1, 2, 3)

            self.assertListEqual(
                [1, 2],
                list(
                    static_node.a1[:],
                ),
                msg="static node a1 should be [1, 2",
            )

    def test_invalid_node_accessing(self):
        for backend_name in backends_to_test:
            frm = build_frame(backend_name=backend_name)

            static_node: StaticNode = frm.static_nodes[0]

            # get attribute value with not supported parameter
            with self.assertRaises(BackendsGetItemInvalidException) as ctx:
                static_node.a1["a"]

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
            self.assertEqual(
                2,
                node_info["static"]["attributes"]["a1"]["slots"],
            )
            self.assertEqual(
                1,
                node_info["static"]["attributes"]["a2"]["slots"],
            )

    def test_enable_snapshots(self):
        for backend_name in backends_to_test:
            """Test if snapshot enabled"""
            frame = build_frame(
                enable_snapshot=True,
                backend_name=backend_name,
            )

            # snapshots should not be None
            self.assertIsNotNone(frame)

            # length should be 0 before taking snapshot
            self.assertEqual(
                0,
                len(frame.snapshots),
                msg="snapshot length should be 0",
            )

            # another frame without snapshots enabled
            frame1 = build_frame(backend_name=backend_name)

            self.assertIsNone(frame1.snapshots)

    def test_reset(self):
        for backend_name in backends_to_test:
            """Test reset work as expected, reset all attributes to 0"""
            frame = build_frame(backend_name=backend_name)

            frame.static_nodes[0].a1[:] = (1, 234)

            # before reset
            self.assertListEqual(
                [1, 234],
                list(
                    frame.static_nodes[0].a1[:],
                ),
                msg="static node's a1 should be [1, 234] before reset",
            )

            frame.reset()

            # after reset
            self.assertListEqual(
                [0, 0],
                list(
                    frame.static_nodes[0].a1[:],
                ),
                msg="static node's a1 should be [0, 0] after reset",
            )

    def test_append_nodes(self):
        # NOTE: this case only support raw backend
        frame = build_frame(
            enable_snapshot=True,
            total_snapshot=10,
            backend_name="dynamic",
        )

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
        self.assertTrue((states[-2:].astype(np.int) == 0).all())

        states = static_snapshot[1::"a3"]

        self.assertTupleEqual(states.shape, (1, 7, 1, 1))

        states = states.flatten()

        self.assertEqual(7, len(states))

        # no padding value
        self.assertListEqual(
            [0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0],
            list(states),
        )

        # with specify node indices, will not padding to max node number
        states = static_snapshot[0:[0, 1, 2, 3, 4]:"a3"]

        self.assertTupleEqual(states.shape, (1, 5, 1, 1))

        self.assertListEqual(
            [0.0, 0.0, 0.0, 0.0, 9.0],
            list(states.flatten()[0:5]),
        )

        frame.snapshots.reset()
        frame.reset()

        # node number will resume to origin one after reset
        self.assertEqual(STATIC_NODE_NUM, len(frame.static_nodes))

    def test_delete_node(self):
        frame = build_frame(
            enable_snapshot=True,
            total_snapshot=10,
            backend_name="dynamic",
        )

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
            second_static_node.a3

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
        self.assertEqual(0, int(states[1]))

        self.assertListEqual(
            [0.0, 0.0, 0.0, 123.0],
            list(states[[0, 2, 3, 4]]),
        )

        # then we resume the deleted node, this mark it as not deleted, but values will be reset to 0
        frame.resume_node(second_static_node)

        # DELETE node's value will not be reset after deleted
        self.assertEqual(444, second_static_node.a3)

        second_static_node.a3 = 222

        frame.take_snapshot(2)

        states = static_snapshots[2::"a3"]

        self.assertTupleEqual(states.shape, (1, 5, 1, 1))

        states = states.flatten()

        self.assertListEqual([0.0, 222.0, 0.0, 0.0, 123.0], list(states))

        frame.snapshots.reset()
        frame.reset()

        # node number will resume to origin one after reset
        self.assertEqual(STATIC_NODE_NUM, len(frame.static_nodes))

        # and no nodes marked as deleted
        for node in frame.static_nodes:
            self.assertTrue(node.is_deleted == False)

    def test_invalid_attribute_description(self):
        # we do not support const list attribute

        @node("test")
        class TestNode(NodeBase):
            a1 = NodeAttribute("i", 2, is_const=True, is_list=True)

        class TestFrame(FrameBase):
            test_nodes = FrameNode(TestNode, 1)

            def __init__(self):
                super().__init__(enable_snapshot=True, total_snapshot=10, backend_name="dynamic")

        with self.assertRaises(RuntimeError) as ctx:
            TestFrame()

    def test_query_const_attribute_without_taking_snapshot(self):
        @node("test")
        class TestNode(NodeBase):
            a1 = NodeAttribute("i", 2, is_const=True)

        class TestFrame(FrameBase):
            test_nodes = FrameNode(TestNode, 2)

            def __init__(self):
                super().__init__(enable_snapshot=True, total_snapshot=10, backend_name="dynamic")

        frame = TestFrame()

        t1 = frame.test_nodes[0]

        t1.a1[0] = 10

        t1_ss = frame.snapshots["test"]

        # default snapshot length is 0
        self.assertEqual(0, len(frame.snapshots))

        # we DO have to provide a tick to it for padding, as there is no snapshots there
        states = t1_ss[0::"a1"]

        states = states.flatten()

        self.assertListEqual([10.0, 0.0, 0.0, 0.0], list(states))

    def test_list_attribute(self):
        @node("test")
        class TestNode(NodeBase):
            a1 = NodeAttribute("i", 1, is_list=True)
            a4 = NodeAttribute("i", 1, is_list=True)
            a2 = NodeAttribute("i", 2, is_const=True)
            a3 = NodeAttribute("i")

        class TestFrame(FrameBase):
            test_nodes = FrameNode(TestNode, 2)

            def __init__(self):
                super().__init__(enable_snapshot=True, total_snapshot=10, backend_name="dynamic")

        frame = TestFrame()

        frame.take_snapshot(0)

        n1 = frame.test_nodes[0]

        n1.a2[:] = (2221, 2222)
        n1.a3 = 333

        # slot number of list attribute is 0 by default
        # so get/set value by index will cause error

        # append value to it
        n1.a1.append(10)
        n1.a1.append(11)
        n1.a1.append(12)

        n1.a4.append(100)
        n1.a4.append(101)

        expected_value = [10, 11, 12]

        # check if value set append correct
        self.assertListEqual(expected_value, n1.a1[:])

        # Check if length correct
        self.assertEqual(3, len(n1.a1))

        # For loop to go through all items in list
        for i, a_value in enumerate(n1.a1):
            self.assertEqual(expected_value[i], a_value)

        frame.take_snapshot(1)

        # resize it to 2
        n1.a1.resize(2)

        # this will cause last value to be removed
        self.assertEqual(2, len(n1.a1))

        self.assertListEqual([10, 11], n1.a1[:])

        # exterd its size, then default value should be 0
        n1.a1.resize(5)

        self.assertEqual(5, len(n1.a1))
        self.assertListEqual([10, 11, 0, 0, 0], n1.a1[:])

        # clear will cause length be 0
        n1.a1.clear()

        self.assertEqual(0, len(n1.a1))

        # insert a new value to 0, as it is empty now
        n1.a1.insert(0, 10)

        self.assertEqual(1, len(n1.a1))

        self.assertEqual(10, n1.a1[0])

        # [11, 10] after insert
        n1.a1.insert(0, 11)

        # remove 2nd one
        n1.a1.remove(1)

        self.assertEqual(1, len(n1.a1))

        self.assertEqual(11, n1.a1[0])

        # test if snapshot correct
        # NOTE: list attribute querying need to provide 1 attribute and 1 node index
        states = frame.snapshots["test"][0:0:"a1"]

        # first tick a1 has no value, so states will be None
        self.assertIsNone(states)

        states = frame.snapshots["test"][1:0:"a1"]
        states = states.flatten()

        # a1 has 3 value at tick 1
        self.assertEqual(3, len(states))

        self.assertListEqual([10, 11, 12], list(states))

        # tick can be empty, then means get state for latest snapshot
        states = frame.snapshots["test"][:0:"a1"].flatten()

        self.assertEqual(3, len(states))
        self.assertListEqual([10, 11, 12], list(states))

        # check states after reset
        frame.reset()
        frame.snapshots.reset()

        # list attribute should be cleared
        self.assertEqual(0, len(n1.a1))
        self.assertEqual(0, len(n1.a4))

        # then append value to each list attribute to test if value will be mixed
        n1.a1.append(10)
        n1.a1.append(20)

        n1.a4.append(100)
        n1.a4.append(200)

        self.assertEqual(10, n1.a1[0])
        self.assertEqual(20, n1.a1[1])
        self.assertEqual(100, n1.a4[0])
        self.assertEqual(200, n1.a4[1])

    def test_list_attribute_with_large_size(self):
        @node("test")
        class TestNode(NodeBase):
            a1 = NodeAttribute("i", 1, is_list=True)

        class TestFrame(FrameBase):
            test_nodes = FrameNode(TestNode, 2)

            def __init__(self):
                super().__init__(backend_name="dynamic")

        frame = TestFrame()

        n1a1 = frame.test_nodes[0].a1

        max_size = 200 * 10000

        for i in range(max_size):
            n1a1.append(1)

        print(len(n1a1))
        self.assertEqual(max_size, len(n1a1))

    def test_list_attribute_invalid_index_access(self):
        @node("test")
        class TestNode(NodeBase):
            a1 = NodeAttribute("i", 1, is_list=True)

        class TestFrame(FrameBase):
            test_nodes = FrameNode(TestNode, 2)

            def __init__(self):
                super().__init__(backend_name="dynamic")

        frame = TestFrame()

        n1a1 = frame.test_nodes[0].a1

        # default list attribute's size is 0, so index accessing will out of range
        with self.assertRaises(RuntimeError) as ctx:
            n1a1[0]

        with self.assertRaises(RuntimeError) as ctx:
            n1a1.remove(0)

    def test_frame_dump(self):
        frame = build_frame(enable_snapshot=True, total_snapshot=10, backend_name="dynamic")

        frame.dump(".")

        # there should be 2 output files
        self.assertTrue(os.path.exists("node_static.csv"))
        self.assertTrue(os.path.exists("node_dynamic.csv"))
        list_parser = lambda c: c if not c.startswith("[") else [float(n) for n in c.strip("[] ,").split(",")]

        # a1 is a list
        static_df = pd.read_csv("node_static.csv", converters={"a1": list_parser})

        # all value should be 0
        for i in range(STATIC_NODE_NUM):
            row = static_df.loc[i]

            a1 = row["a1"]
            a2 = row["a2"]
            a3 = row["a3"]

            self.assertEqual(2, len(a1))

            self.assertListEqual([0.0, 0.0], a1)
            self.assertEqual(0, a2)
            self.assertEqual(0, a3)

        frame.take_snapshot(0)

        frame.take_snapshot(1)

        frame.snapshots.dump(".")

        self.assertTrue(os.path.exists("snapshots_dynamic.csv"))
        self.assertTrue(os.path.exists("snapshots_static.csv"))

    def test_frame_attribute_filtering(self):
        batch_number = 100

        for backend_name in backends_to_test:
            print("current backend:", backend_name)

            @node("test")
            class TestNode(NodeBase):
                a1 = NodeAttribute("i", batch_number)
                a2 = NodeAttribute("i")

            class TestFrame(FrameBase):
                test_nodes = FrameNode(TestNode, 2)

                def __init__(self):
                    super().__init__(enable_snapshot=True, total_snapshot=10, backend_name=backend_name)

            frame = TestFrame()

            node1 = frame.test_nodes[0]

            # initial value
            node1.a1[:] = [i for i in range(batch_number)]

            results = node1.a1.where(lambda x: x < 10)

            # value of index 0...9 match < 10 filter
            self.assertListEqual([i for i in range(10)], results)

            # no matched one
            results = node1.a1.where(lambda x: x > batch_number)

            self.assertEqual(0, len(results))

            # use basic comparison
            results = node1.a1 < 10

            self.assertListEqual([i for i in range(10)], results)

            results = node1.a1 > batch_number

            self.assertEqual(0, len(results))

            results = node1.a1 == 10

            self.assertEqual(10, results[0])

            results = node1.a1 <= 10

            self.assertListEqual([i for i in range(10 + 1)], results)

            results = node1.a1 >= 99

            self.assertEqual(1, len(results))

            results = node1.a1 != 99

            self.assertEqual(99, len(results))

            self.assertListEqual([i for i in range(99)], results)


if __name__ == "__main__":
    unittest.main()
