# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import numpy as np

from maro.rl import UnboundedStore, FixedSizeStore, OverwriteType


class TestUnboundedStore(unittest.TestCase):
    def test_put(self):
        store = UnboundedStore()
        indexes = store.put([[1, -1], [2, -2], [3, -3]])
        expected = [0, 1, 2]
        self.assertEqual(indexes, expected, msg=f"expected returned indexes = {expected}, got {indexes}")
        indexes = store.put([[4, -4], [5, -5]])
        expected = [3, 4]
        self.assertEqual(indexes, expected, msg=f"expected returned indexes = {expected}, got {indexes}")

    def test_get(self):
        store = UnboundedStore()
        store.put([[1, -1], [2, -2], [3, -3], [4, -4], [5, -5]])
        indexes = [1, 4]
        actual = store.get(indexes)
        expected = [[2, -2], [5, -5]]
        self.assertEqual(actual, expected, msg=f"expected {expected}, got {actual}")

    def test_update(self):
        store = UnboundedStore()
        store.put([{"a": 1, "b": 6}, {"a": 2, "b": 3}, {"a": 3, "b": 0}, {"a": 4, "b": 9}, {"a": 5, "b": 3}])
        store.update([0, 3], [{"a": -1, "b": -6}, {"a": -4, "b": -9}])
        actual = store.take()
        expected = [{"a": -1, "b": -6}, {"a": 2, "b": 3}, {"a": 3, "b": 0}, {"a": -4, "b": -9}, {"a": 5, "b": 3}]
        self.assertEqual(actual, expected, msg=f"expected store content = {expected}, got {actual}")
        store.update([1, 2], [7, 8], key="b")
        actual = store.take()
        expected = [{"a": -1, "b": -6}, {"a": 2, "b": 7}, {"a": 3, "b": 8}, {"a": -4, "b": -9}, {"a": 5, "b": 3}]
        self.assertEqual(actual, expected, msg=f"expected store content = {expected}, got {actual}")

    def test_filter(self):
        store = UnboundedStore()
        store.put([{"a": 1, "b": 6}, {"a": 2, "b": 3}, {"a": 3, "b": 9}, {"a": 4, "b": 8}, {"a": 5, "b": 3}])
        result = store.apply_multi_filters(filters=[lambda x: x["a"] > 2, lambda x: x["b"] & 1 != 0])
        expected = [{"a": 3, "b": 9}, {"a": 5, "b": 3}]
        self.assertEqual(result, expected, msg=f"expected {expected}, got {result}")


class TestFixedSizeStore(unittest.TestCase):
    def test_put_with_rolling_overwrite(self):
        store = FixedSizeStore(capacity=5, overwrite_type=OverwriteType.ROLLING)
        indexes = store.put([[1, -1], [2, -2], [3, -3]])
        expected = np.array([0, 1, 2])
        self.assertTrue(np.array_equal(indexes, expected), msg=f"expected indexes = {expected}, got {indexes}")
        indexes = store.put([[4, -4], [5, -5], [6, -6], [7, -7]])
        expected = np.array([-2, -1, 0, 1])
        self.assertTrue(np.array_equal(indexes, expected), msg=f"expected indexes = {expected}, got {indexes}")
        actual = np.vstack(store.take())
        expected = np.array([[6, -6], [7, -7], [3, -3], [4, -4], [5, -5]])
        self.assertTrue(np.array_equal(actual, expected), msg=f"expected store content = {expected}, got {actual}")

    def test_put_in_fixed_size_store_with_random_overwrite(self):
        store = FixedSizeStore(capacity=5, overwrite_type=OverwriteType.RANDOM)
        indexes_1 = list(store.put([[1, -1], [2, -2], [3, -3]]))
        indexes_2 = store.put([[4, -4], [5, -5], [6, -6], [7, -7]])
        for i in indexes_2[2:]:
            self.assertIn(i, indexes_1, msg=f"expected overwrite index in {indexes_1}, got {i}")

    def test_get(self):
        store = FixedSizeStore(capacity=5, overwrite_type=OverwriteType.RANDOM)
        store.put([[1, -1], [2, -2], [3, -3], [4, -4], [5, -5]])
        indexes = [1, 4]
        actual = np.vstack(store.get(indexes))
        expected = np.array([[2, -2], [5, -5]])
        self.assertTrue(np.array_equal(actual, expected), msg=f"expected {expected}, got {actual}")

    def test_update(self):
        store = FixedSizeStore(capacity=5, overwrite_type=OverwriteType.ROLLING)
        store.put([{"a": 1, "b": 6}, {"a": 2, "b": 3}, {"a": 3, "b": 0}, {"a": 4, "b": 9}, {"a": 5, "b": 3}])
        store.update([0, 3], [{"a": -1, "b": -6}, {"a": -4, "b": -9}])
        actual = store.take()
        expected = np.array([{"a": -1, "b": -6}, {"a": 2, "b": 3}, {"a": 3, "b": 0},
                             {"a": -4, "b": -9}, {"a": 5, "b": 3}])
        self.assertTrue(np.array_equal(actual, expected), msg=f"expected store content = {expected}, got {actual}")
        store.update([1, 2], [7, 8], key="b")
        actual = store.take()
        expected = np.array([{"a": -1, "b": -6}, {"a": 2, "b": 7}, {"a": 3, "b": 8},
                             {"a": -4, "b": -9}, {"a": 5, "b": 3}])
        self.assertTrue(np.array_equal(actual, expected), msg=f"expected store content = {expected}, got {actual}")

    def test_filter(self):
        store = FixedSizeStore(capacity=5, overwrite_type=OverwriteType.ROLLING)
        store.put([{"a": 1, "b": 6}, {"a": 2, "b": 3}, {"a": 3, "b": 9}, {"a": 4, "b": 8}, {"a": 5, "b": 3}])
        result = store.apply_multi_filters(filters=[lambda x: x["a"] > 2, lambda x: x["b"] & 1 != 0])
        expected = np.array([{"a": 3, "b": 9}, {"a": 5, "b": 3}])
        self.assertTrue(np.array_equal(result, expected), msg=f"expected {expected}, got {result}")

if __name__ == "__main__":
    unittest.main()
