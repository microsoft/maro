# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import numpy as np

from maro.rl.utils.trajectory_utils import get_k_step_returns, get_lambda_returns


class TestTrajectoryUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.rewards = np.asarray([3, 2, 4, 1, 5])
        self.values = np.asarray([4, 7, 1, 3, 6])
        self.lam = 0.6
        self.discount = 0.8
        self.k = 4

    def test_k_step_return(self):
        returns = get_k_step_returns(self.rewards, self.values, self.discount, k=self.k)
        expected = np.asarray([10.1296, 8.912, 8.64, 5.8, 6.0])
        np.testing.assert_allclose(returns, expected, rtol=1e-4)

    def test_lambda_return(self):
        returns = get_lambda_returns(self.rewards, self.values, self.discount, self.lam, k=self.k)
        expected = np.asarray([8.1378176, 6.03712, 7.744, 5.8, 6.0])
        np.testing.assert_allclose(returns, expected, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
