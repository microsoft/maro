# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import unittest
from maro.simulator.core import Env, BusinessEngineNotFoundError

MAX_TICK = 10


def run_to_end(env: Env):
    _, _, is_done = env.step(None)

    while not is_done:
        _, _, is_done = env.step(None)


class TestEnv(unittest.TestCase):
    """
    this test will use dummy scenario
    """

    def setUp(self) -> None:
        self.env = Env("ecr", "5p_ssddd", 10)

    def test_env_correct(self):
        run_to_end(self.env)

        # check if the snapshot number equals with max_tick
        num_of_snapshots = len(self.env.snapshot_list)

        self.assertEqual(MAX_TICK, num_of_snapshots, msg=f"number of snapshots ({num_of_snapshots}) should be same "
                                                         f"with max tick ({MAX_TICK})")

        # check if the max tick, [0, MAX_TICK)
        self.assertEqual(MAX_TICK-1, self.env.tick)

    def test_reset(self):
        run_to_end(self.env)

        self.env.reset()

        num_of_snapshots = len(self.env.snapshot_list)

        # NOTE: this should be part of scenario
        self.assertEqual(0, num_of_snapshots, msg=f"number of snapshots ({num_of_snapshots}) should be 0 after reset")
        self.assertEqual(0, self.env.tick, msg=f"tick ({self.env.tick}) should be 0 after reset")

    def test_invalid_scenario(self):
        with self.assertRaises(ModuleNotFoundError) as ctx:
            env = Env("None", "5p_ssddd", 100)

        with self.assertRaises(AssertionError) as ctx:
            env = Env("dummy", "None", 0)



if __name__ == "__main__":
    unittest.main()
