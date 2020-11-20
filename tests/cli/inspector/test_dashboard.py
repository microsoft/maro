import unittest

from maro.cli.inspector.common_helper import holder_sample_ratio, get_snapshot_sample


class TestDashboard(unittest.TestCase):
    """ Test calculation results in inspector-dashboard"""
    def test_get_holder_sample_ratio(self):
        """Test method holder_sample_ratio(a)"""
        self.assertEqual(
            [0.1667, 0.3334, 0.5001, 0.6668, 0.8335, 1],
            holder_sample_ratio(6)
        )
        self.assertNotEqual(
            [0.2, 0.4, 0.6, 0.8, 1],
            holder_sample_ratio(5)
        )

    def test_get_snapshot_sample(self):
        """Test method get_sanpshot_sample(a,b)"""
        self.assertEqual(
            [
                0, 1, 3, 5, 7, 9, 11,
                13, 15, 17, 19, 21, 23,
                25, 27, 29, 31, 33, 35,
                37, 39, 41, 43, 45, 47,
                49, 51, 53, 55, 57, 59,
                61, 63, 65, 67, 69, 71,
                73, 75, 77, 79, 81, 83,
                85, 87, 89, 91, 93, 95,
                97, 99, 100
            ],
            get_snapshot_sample(101, 0.459)
        )
        self.assertNotEqual(
            [0, 1, 100],
            get_snapshot_sample(101, 0.0099)
        )


if __name__ == '__main__':
    unittest.main()
