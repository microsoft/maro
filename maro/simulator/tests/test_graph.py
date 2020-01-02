# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import unittest

from maro.simulator.graph import Graph, GraphAttributeType, ResourceNodeType, GraphAttributeNotFoundError, GraphAlreadySetupError

STATIC_NODE_NUM = 10
DYNAMIC_NODE_NUM = 10

class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(STATIC_NODE_NUM, DYNAMIC_NODE_NUM)

        self.graph.register_attribute("a1", GraphAttributeType.FLOAT, 1)
        self.graph.register_attribute("a2", GraphAttributeType.INT, 1)
        self.graph.register_attribute("m1", GraphAttributeType.INT_MAT, 50, 5, 10)
        self.graph.register_attribute("m2", GraphAttributeType.INT_MAT, 10, 1, 10)

        self.graph.setup()

    def test_graph_properties(self):
        self.assertEqual(self.graph.static_node_number, STATIC_NODE_NUM)
        self.assertEqual(self.graph.dynamic_node_number, DYNAMIC_NODE_NUM)

    def test_get_attribute(self):
        # the initialize value is 0
        for i in range(DYNAMIC_NODE_NUM):
            v = self.graph.get_attribute(ResourceNodeType.DYNAMIC, i, 'a1', 0)
            
            self.assertEqual(v, 0)

        for i in range(STATIC_NODE_NUM):
            v = self.graph.get_attribute(ResourceNodeType.STATIC, i, "a1", 0)

            self.assertEqual(v, 0)

        # get matrix value, it should be 0
        for i in range(5):
            for j in range(10):
                v = self.graph.get_int_matrix_value("m1", i, j)

                self.assertEqual(v, 0)

        for i in range(10):
            v = self.graph.get_int_matrix_value("m2", 0, i)

            self.assertEqual(v, 0)

    def test_get_attribute_without_registered(self):
        with self.assertRaises(GraphAttributeNotFoundError) as ctx:
            self.graph.get_attribute(ResourceNodeType.STATIC, 0, "a3", 0)

    def test_set_attribute_without_registered(self):
        with self.assertRaises(GraphAttributeNotFoundError) as ctx:
            self.graph.get_attribute(ResourceNodeType.STATIC, 0, "a3", 0)

    def test_set_attribute_none(self):
        with self.assertRaises(TypeError) as ctx:
            self.graph.set_attribute(ResourceNodeType.DYNAMIC, 0, "a1", 0, None)

        with self.assertRaises(TypeError) as ctx:
            self.graph.get_attribute(None, 0, "a1", 0)

        with self.assertRaises(GraphAttributeNotFoundError) as ctx:
            self.graph.get_attribute(ResourceNodeType.DYNAMIC, 0, None, 0)

    def test_get_set_attribute(self):
        self.graph.set_attribute(ResourceNodeType.STATIC, 0, "a1", 0, 12)

        v = 0

        # the initialize value is 0, the dynamic nodes should not be changed
        for i in range(DYNAMIC_NODE_NUM):
            v = self.graph.get_attribute(ResourceNodeType.DYNAMIC, i, 'a1', 0)
            
            self.assertEqual(v, 0)

        # only first static node has value 12
        for i in range(STATIC_NODE_NUM):
            v = self.graph.get_attribute(ResourceNodeType.STATIC, i, "a1", 0)

            if i != 0:
                self.assertEqual(v, 0)
            else:
                self.assertEqual(v, 12)

        # check if there is any conflict 
        for i in range(STATIC_NODE_NUM):
            self.graph.set_attribute(ResourceNodeType.STATIC, i, "a1", 0, i * 10 + 1)

        for i in range(STATIC_NODE_NUM):
            v = self.graph.get_attribute(ResourceNodeType.STATIC, i, "a1", 0)

            self.assertEqual(v, i * 10 + 1)
        
        for i in range(DYNAMIC_NODE_NUM):
            v = self.graph.get_attribute(ResourceNodeType.DYNAMIC, i, "a1", 0)

            self.assertNotEqual(v, i * 10 + 1)

        # set value for matrix, and check result
        for i in range(5):
            for j in range(10):
                self.graph.set_int_matrix_value("m1", i, j, i*10+j)

        for i in range(5):
            for j in range(10):
                v = self.graph.get_int_matrix_value("m1", i, j)

                self.assertEqual(i*10+j, v)

        # check no conflict
        for i in range(10):
            v = self.graph.get_int_matrix_value("m2", 0, i)

            self.assertEqual(0, v)

    def test_register_after_setup(self):
        # this will not register any attribute as we already setup
        self.graph.register_attribute("aa", GraphAttributeType.FLOAT, 12)

        with self.assertRaises(GraphAttributeNotFoundError) as ctx:
            self.graph.get_attribute(ResourceNodeType.DYNAMIC, 0, None, 0)

    def test_reset(self):
        self.graph.set_attribute(ResourceNodeType.STATIC, 0, "a1", 0, 12)

        self.graph.reset()

        v = self.graph.get_attribute(ResourceNodeType.STATIC, 0, "a1", 0)

        self.assertEqual(0, v)


if __name__ == "__main__":
    unittest.main()

    