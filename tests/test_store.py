# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

from maro.rl import ExperienceMemory


class TestUnboundedStore(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ExperienceMemory(["a", "b", "c"])

    def tearDown(self) -> None:
        self.store.clear()

    def test_put(self):
        indexes = self.store.put({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        expected = [0, 1, 2]
        self.assertEqual(indexes, expected, msg=f"expected returned indexes = {expected}, got {indexes}")
        indexes = self.store.put({"a": [10, 11], "b": [12, 13], "c": [14, 15]})
        expected = [3, 4]
        self.assertEqual(indexes, expected, msg=f"expected returned indexes = {expected}, got {indexes}")

    def test_get(self):
        self.store.put({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [9, 10, 11, 12]})
        indexes = [1, 3]
        actual = self.store.get(indexes)
        expected = {"a": [2, 4], "b": [6, 8], "c": [10, 12]}
        self.assertEqual(actual, expected, msg=f"expected {expected}, got {actual}")

    def test_update(self):
        self.store.put({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10], "c": [11, 12, 13, 14, 15]})
        self.store.update([0, 3], {"a": [-1, -4], "c": [-11, -14]})
        actual = self.store.dumps()
        expected = {"a": [-1, 2, 3, -4, 5], "b": [6, 7, 8, 9, 10], "c": [-11, 12, 13, -14, 15]}
        self.assertEqual(actual, expected, msg=f"expected store content = {expected}, got {actual}")

    def test_filter(self):
        self.store.put({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10], "c": [11, 12, 13, 14, 15]})
        result = self.store.apply_multi_filters(filters=[lambda x: x["a"] > 2, lambda x: sum(x.values()) % 2 == 0])[1]
        expected = {"a": [3, 5], "b": [8, 10], "c": [13, 15]}
        self.assertEqual(result, expected, msg=f"expected {expected}, got {result}")


class TestFixedSizeStore(unittest.TestCase):
    def test_put_with_rolling_overwrite(self):
        store = ExperienceMemory(["a", "b", "c"], capacity=5, overwrite_type="rolling")
        indexes = store.put({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        expected = [0, 1, 2]
        self.assertEqual(indexes, expected, msg=f"expected indexes = {expected}, got {indexes}")
        indexes = store.put({"a": [10, 11, 12, 13], "b": [14, 15, 16, 17], "c": [18, 19, 20, 21]})
        expected = [-2, -1, 0, 1]
        self.assertEqual(indexes, expected, msg=f"expected indexes = {expected}, got {indexes}")
        actual = store.dumps()
        expected = {"a": [12, 13, 3, 10, 11], "b": [16, 17, 6, 14, 15], "c": [20, 21, 9, 18, 19]}
        self.assertEqual(actual, expected, msg=f"expected store content = {expected}, got {actual}")

    def test_put_with_random_overwrite(self):
        store = ExperienceMemory(["a", "b", "c"], capacity=5, overwrite_type="random")
        indexes = store.put({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        indexes_2 = store.put({"a": [10, 11, 12, 13], "b": [14, 15, 16, 17], "c": [18, 19, 20, 21]})
        for i in indexes_2[2:]:
            self.assertIn(i, indexes, msg=f"expected overwrite index in {indexes}, got {i}")


if __name__ == "__main__":
    unittest.main()
