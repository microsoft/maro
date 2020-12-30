import unittest

from maro.cli.inspector.dashboard_helper import get_sample_ratio_selection_list, get_sample_index_list


class TestDashboard(unittest.TestCase):
    """ Test calculation results in inspector-dashboard"""
    def test_get_sample_ratio_selection_list(self):
        """Test method get_sample_ratio_selection_list(a)"""
        self.assertEqual(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            get_sample_ratio_selection_list(5)
        )
        self.assertNotEqual(
            [0.0, 1.0],
            get_sample_ratio_selection_list(1)
        )

    def test_get_sample_index_list(self):
        """Test method get_sample_index_list(a,b)"""
        self.assertEqual(
            [
                0, 4, 8, 12, 16, 20,
                24, 28, 32, 36, 40,
                44, 48, 52, 56, 60,
                64, 68, 72, 76, 80,
                84, 88, 92, 96
            ],
            get_sample_index_list(100, 0.21)
        )
        self.assertNotEqual(
            [],
            get_sample_index_list(100, 0)
        )


if __name__ == '__main__':
    unittest.main()
