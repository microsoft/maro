# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import unittest

from maro.simulator.graph import Graph, GraphDataType, GraphAttributeType, GraphAttribute

STATIC_NODE_NUM = 10
DYNAMIC_NODE_NUM = 10
DYNAMIC_NODE = GraphAttributeType.DYNAMIC_NODE
STATIC_NODE = GraphAttributeType.STATIC_NODE
GENERAL = GraphAttributeType.GENERAL

class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(STATIC_NODE_NUM, DYNAMIC_NODE_NUM)

        self.graph.register_attribute(DYNAMIC_NODE, "a1", GraphDataType.INT32, 1)
        self.graph.register_attribute(STATIC_NODE, "a2", GraphDataType.INT32, 1)
        self.graph.register_attribute(GENERAL, "m1", GraphDataType.INT32, 50)
        self.graph.register_attribute(GENERAL, "m2", GraphDataType.INT32, 10)

        self.graph.setup()

    def test_graph_properties(self):
        self.assertEqual(self.graph.static_node_number, STATIC_NODE_NUM)
        self.assertEqual(self.graph.dynamic_node_number, DYNAMIC_NODE_NUM)

    def test_get_attribute(self):
        # the initialize value is 0
        for i in range(DYNAMIC_NODE_NUM):
            v = self.graph.get_attribute(DYNAMIC_NODE, i, 'a1', 0)
            
            self.assertEqual(v, 0)

        for i in range(STATIC_NODE_NUM):
            v = self.graph.get_attribute(STATIC_NODE, i, "a2", 0)

            self.assertEqual(v, 0)

        # get matrix value, it should be 0
        for i in range(5):
            for j in range(10):
                v = self.graph.get_attribute(GENERAL, 0, "m1", i*10 + j)

                self.assertEqual(v, 0)

        for i in range(10):
            v = self.graph.get_attribute(GENERAL, 0, "m2", i)

            self.assertEqual(v, 0)

    def _test_get_attribute_without_registered(self):
        with self.assertRaises(GraphAttributeNotFoundError) as ctx:
            self.graph.get_attribute(ResourceNodeType.STATIC, 0, "a3", 0)

    def _test_set_attribute_without_registered(self):
        with self.assertRaises(GraphAttributeNotFoundError) as ctx:
            self.graph.get_attribute(ResourceNodeType.STATIC, 0, "a3", 0)

    def _test_set_attribute_none(self):
        with self.assertRaises(TypeError) as ctx:
            self.graph.set_attribute(ResourceNodeType.DYNAMIC, 0, "a1", 0, None)

        with self.assertRaises(TypeError) as ctx:
            self.graph.get_attribute(None, 0, "a1", 0)

        with self.assertRaises(GraphAttributeNotFoundError) as ctx:
            self.graph.get_attribute(ResourceNodeType.DYNAMIC, 0, None, 0)

    def test_get_set_attribute(self):
        self.graph.set_attribute(STATIC_NODE, 0, "a2", 0, 12)

        v = 0

        # the initialize value is 0, the dynamic nodes should not be changed
        for i in range(DYNAMIC_NODE_NUM):
            v = self.graph.get_attribute(DYNAMIC_NODE, i, 'a1', 0)
            
            self.assertEqual(v, 0)

        # check if setting value corret for each node
        for i in range(STATIC_NODE_NUM):
            self.graph.set_attribute(STATIC_NODE, i, "a2", 0, i * 10 + 1)

        for i in range(STATIC_NODE_NUM):
            v = self.graph.get_attribute(STATIC_NODE, i, "a2", 0)

            self.assertEqual(v, i * 10 + 1)

        # set value for matrix, and check result
        for i in range(5):
            for j in range(10):
                self.graph.set_attribute(GENERAL, 0, "m1", i*10+j, i*10+j)

        for i in range(5):
            for j in range(10):
                v = self.graph.get_attribute(GENERAL, 0, "m1", i*10+j)

                self.assertEqual(i*10+j, v)

        # check no conflict
        for i in range(10):
            v = self.graph.get_attribute(GENERAL, 0, "m2", i)

            self.assertEqual(0, v)

    def _test_register_after_setup(self):
        # this will not register any attribute as we already setup
        self.graph.register_attribute("aa", GraphAttributeType.FLOAT, 12)

        with self.assertRaises(GraphAttributeNotFoundError) as ctx:
            self.graph.get_attribute(ResourceNodeType.DYNAMIC, 0, None, 0)

    def test_dump(self):
        self.graph.set_attribute(STATIC_NODE, 0, "a2", 0, 12)
        
        self.graph.save(b"graph.dat")

    def test_reset(self):
        self.graph.set_attribute(STATIC_NODE, 0, "a2", 0, 12)

        self.graph.reset()

        v = self.graph.get_attribute(STATIC_NODE, 0, "a2", 0)

        self.assertEqual(0, v)


if __name__ == "__main__":
    unittest.main()

    