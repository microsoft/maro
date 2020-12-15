# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import timeit
import unittest

from maro.cli.data_pipeline.data_process import generate, list_env
from maro.simulator.scenarios.vm_scheduling import CpuReader

def test_data_pipeline():
    list_env()
    generate("vm_scheduling", "azure.2019.10k")

class CpuReaderTest(unittest.TestCase):
    def setUp(self):
        data_path = "/mnt/d/kuanwei/data/test_10k/build/vm_cpu_readings-file-1-of-195.bin"
        self.cpu_reader = CpuReader(data_path)

    def test_first_tick(self):
        cpu_utilization_dict = self.cpu_reader.items(tick=0)
        expected = 833
        self.assertEqual(expected, len(cpu_utilization_dict))

if __name__ == "__main__":
    unittest.main()
