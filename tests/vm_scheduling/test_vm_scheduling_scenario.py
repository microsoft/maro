# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

from maro.data_lib import BinaryConverter
from maro.simulator.scenarios.vm_scheduling import CpuReader


def build_binary_file():
    bin_file_name = "tests/data/vm_scheduling/vm_cpu_readings-file-3-of-test.bin"
    meta_file = "tests/data/vm_scheduling/cpu_readings.yml"
    csv_file = "tests/data/vm_scheduling/vm_cpu_readings-file-3-of-test.csv"

    converter = BinaryConverter(bin_file_name, meta_file)
    converter.add_csv(csv_file)
    converter.flush()


class CpuReaderTest(unittest.TestCase):
    def setUp(self):
        data_path = "tests/data/vm_scheduling/vm_cpu_readings-file-1-of-test.bin"
        self.cpu_reader = CpuReader(data_path, 0)

    def tearDown(self):
        self.cpu_reader.reset()

    def test_first_file_first_tick(self):
        cpu_utilization_dict = self.cpu_reader.items(tick=0)
        expected = 4
        self.assertEqual(expected, len(cpu_utilization_dict))

    def test_first_file_last_tick(self):
        cpu_utilization_dict = self.cpu_reader.items(tick=1)
        expected = 13
        self.assertEqual(expected, len(cpu_utilization_dict))

    def test_switch_file(self):
        cpu_utilization_dict = self.cpu_reader.items(tick=1)
        cpu_utilization_dict = self.cpu_reader.items(tick=2)
        expected = 8
        self.assertEqual(expected, len(cpu_utilization_dict))

    def test_last_file(self):
        cpu_utilization_dict = {}
        for i in range(3):
            cpu_utilization_dict = self.cpu_reader.items(tick=i)

        expected = 8
        self.assertEqual(expected, len(cpu_utilization_dict))

        cpu_utilization_dict = self.cpu_reader.items(tick=3)
        expected = 7
        self.assertEqual(expected, len(cpu_utilization_dict))


if __name__ == "__main__":
    unittest.main()
