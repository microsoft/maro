# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import unittest

from maro.vector_env import VectorEnv

from .utils import backends_to_test


class TestVectorEnv(unittest.TestCase):
    def test_batch_number_large_than_cpu_core(self):
        actual_core = os.cpu_count()

        # We pass more batch number, it should cause assert error
        with self.assertRaises(AssertionError) as ctx:
            ve = VectorEnv(
                actual_core + 5,
                scenario="cim",
                topology="toy.5p_ssddd_l0.0",
                durations=100,
            )

    def test_push_single_env(self):
        for backend_name in backends_to_test:
            os.environ["DEFAULT_BACKEND_NAME"] = backend_name
            with VectorEnv(
                batch_num=2,
                scenario="cim",
                topology="toy.5p_ssddd_l0.0",
                durations=100,
            ) as ve:

                # We use dict to specified which environment should go to next step.

                # Push 1st environment to next step
                metrics, decision_event, is_done = ve.step({0: None})

                # ticks of each environment
                ticks = ve.tick

                # tick of 1st environment should be greater than 0, 2nd should being 0, as it is not being pushed
                self.assertEqual(0, ticks[1])
                self.assertGreater(ticks[0], 0)

                # push both 2 environment by pass None action, which will be broadcast to all environment
                while not is_done:
                    metrics, decision_event, is_done = ve.step(None)

                # The ticks should be same at the end
                ticks = ve.tick
                self.assertListEqual([99, 99], ticks)


if __name__ == "__main__":
    unittest.main()
