# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

from maro.cli.inspector.dashboard_helper import get_sample_index_list, get_sample_ratio_selection_list


class TestDashboard(unittest.TestCase):
    """Test calculation results in inspector-dashboard"""

    def test_get_sample_ratio_selection_list(self):
        """Test method get_sample_ratio_selection_list(a)"""
        self.assertEqual(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            get_sample_ratio_selection_list(5),
        )

    def test_get_sample_index_list(self):
        """Test method get_sample_index_list(a,b)"""
        self.assertEqual(
            [0, 10, 20, 30, 40],
            get_sample_index_list(50, 0.1),
        )


if __name__ == "__main__":
    unittest.main()
