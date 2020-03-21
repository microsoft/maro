# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import unittest

from maro.simulator.frame import (Frame, FrameAlreadySetupError,
                                  FrameAttributeNotFoundError,
                                  FrameAttributeType, FrameNodeType)

STATIC_NODE_NUM = 10
DYNAMIC_NODE_NUM = 10

class TestFrame(unittest.TestCase):
    def setUp(self):
        self.frame = Frame(STATIC_NODE_NUM, DYNAMIC_NODE_NUM)

        self.frame.register_attribute("a1", FrameAttributeType.FLOAT, 1)
        self.frame.register_attribute("a2", FrameAttributeType.INT, 1)
        self.frame.register_attribute("m1", FrameAttributeType.INT_MAT, 50, 5, 10)
        self.frame.register_attribute("m2", FrameAttributeType.INT_MAT, 10, 1, 10)

        self.frame.setup()

    def test_frame_properties(self):
        self.assertEqual(self.frame.static_node_number, STATIC_NODE_NUM)
        self.assertEqual(self.frame.dynamic_node_number, DYNAMIC_NODE_NUM)

    def test_get_attribute(self):
        # the initialize value is 0
        for i in range(DYNAMIC_NODE_NUM):
            v = self.frame.get_attribute(FrameNodeType.DYNAMIC, i, 'a1', 0)
            
            self.assertEqual(v, 0)

        for i in range(STATIC_NODE_NUM):
            v = self.frame.get_attribute(FrameNodeType.STATIC, i, "a1", 0)

            self.assertEqual(v, 0)

        # get matrix value, it should be 0
        for i in range(5):
            for j in range(10):
                v = self.frame.get_int_matrix_value("m1", i, j)

                self.assertEqual(v, 0)

        for i in range(10):
            v = self.frame.get_int_matrix_value("m2", 0, i)

            self.assertEqual(v, 0)

    def test_get_attribute_without_registered(self):
        with self.assertRaises(FrameAttributeNotFoundError) as ctx:
            self.frame.get_attribute(FrameNodeType.STATIC, 0, "a3", 0)

    def test_set_attribute_without_registered(self):
        with self.assertRaises(FrameAttributeNotFoundError) as ctx:
            self.frame.get_attribute(FrameNodeType.STATIC, 0, "a3", 0)

    def test_set_attribute_none(self):
        with self.assertRaises(TypeError) as ctx:
            self.frame.set_attribute(FrameNodeType.DYNAMIC, 0, "a1", 0, None)

        with self.assertRaises(TypeError) as ctx:
            self.frame.get_attribute(None, 0, "a1", 0)

        with self.assertRaises(FrameAttributeNotFoundError) as ctx:
            self.frame.get_attribute(FrameNodeType.DYNAMIC, 0, None, 0)

    def test_get_set_attribute(self):
        self.frame.set_attribute(FrameNodeType.STATIC, 0, "a1", 0, 12)

        v = 0

        # the initialize value is 0, the dynamic nodes should not be changed
        for i in range(DYNAMIC_NODE_NUM):
            v = self.frame.get_attribute(FrameNodeType.DYNAMIC, i, 'a1', 0)
            
            self.assertEqual(v, 0)

        # only first static node has value 12
        for i in range(STATIC_NODE_NUM):
            v = self.frame.get_attribute(FrameNodeType.STATIC, i, "a1", 0)

            if i != 0:
                self.assertEqual(v, 0)
            else:
                self.assertEqual(v, 12)

        # check if there is any conflict 
        for i in range(STATIC_NODE_NUM):
            self.frame.set_attribute(FrameNodeType.STATIC, i, "a1", 0, i * 10 + 1)

        for i in range(STATIC_NODE_NUM):
            v = self.frame.get_attribute(FrameNodeType.STATIC, i, "a1", 0)

            self.assertEqual(v, i * 10 + 1)
        
        for i in range(DYNAMIC_NODE_NUM):
            v = self.frame.get_attribute(FrameNodeType.DYNAMIC, i, "a1", 0)

            self.assertNotEqual(v, i * 10 + 1)

        # set value for matrix, and check result
        for i in range(5):
            for j in range(10):
                self.frame.set_int_matrix_value("m1", i, j, i*10+j)

        for i in range(5):
            for j in range(10):
                v = self.frame.get_int_matrix_value("m1", i, j)

                self.assertEqual(i*10+j, v)

        # check no conflict
        for i in range(10):
            v = self.frame.get_int_matrix_value("m2", 0, i)

            self.assertEqual(0, v)

    def test_register_after_setup(self):
        # this will not register any attribute as we already setup
        self.frame.register_attribute("aa", FrameAttributeType.FLOAT, 12)

        with self.assertRaises(FrameAttributeNotFoundError) as ctx:
            self.frame.get_attribute(FrameNodeType.DYNAMIC, 0, None, 0)

    def test_reset(self):
        self.frame.set_attribute(FrameNodeType.STATIC, 0, "a1", 0, 12)

        self.frame.reset()

        v = self.frame.get_attribute(FrameNodeType.STATIC, 0, "a1", 0)

        self.assertEqual(0, v)


if __name__ == "__main__":
    unittest.main()
