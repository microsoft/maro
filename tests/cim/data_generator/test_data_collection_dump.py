# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
import unittest

import numpy as np

from maro.data_lib.cim.cim_data_dump import dump_from_config

MAX_TICK = 20

class TestDataCollectionDump(unittest.TestCase):
    def test_dump(self):
        output_folder = tempfile.mkdtemp()
        config_path = os.path.join("tests", "data", "cim", "data_generator", "dumps", "config.yml")

        dump_from_config(config_path, output_folder, 20)

        # check output folder
        for fname in ["ports.csv", "vessels.csv", "routes.csv", "global_order_proportion.txt", "order_proportion.csv", "misc.yml"]:
            self.assertTrue(os.path.exists(os.path.join(output_folder, fname)), fname)

        # TODO: check content?

    def test_dump_with_invalid_parameters(self):
        output_folder = tempfile.mkdtemp()
        config_path = os.path.join("data", "cim", "data_generator", "dumps", "config.yml")

        with self.assertRaises(AssertionError):
            dump_from_config(None, output_folder, 20)

        with self.assertRaises(AssertionError):
            dump_from_config("", output_folder, 20)

        with self.assertRaises(AssertionError):
            dump_from_config(config_path, None, 20)

if __name__=="__main__":
    unittest.main()
