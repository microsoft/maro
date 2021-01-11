# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

from maro.data_lib import BinaryConverter
from maro.simulator.scenarios.vm_scheduling import CpuReader


class CpuReaderTest(unittest.TestCase):

    for i in range(1, 4):
        meta_file = "tests/data/vm_scheduling/cpu_readings.yml"
        bin_file_name = f"tests/data/vm_scheduling/vm_cpu_readings-file-{i}-of-test.bin"
        csv_file = f"tests/data/vm_scheduling/vm_cpu_readings-file-{i}-of-test.csv"

        converter = BinaryConverter(bin_file_name, meta_file)
        converter.add_csv(csv_file)
        converter.flush()

    data_path = "tests/data/vm_scheduling/vm_cpu_readings-file-1-of-test.bin"

    def setUp(self):
        self.cpu_reader = CpuReader(self.data_path, 0)

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

    def test_reset(self):
        self.cpu_reader.items(tick=0)
        self.cpu_reader.items(tick=1)
        self.cpu_reader.items(tick=2)
        self.cpu_reader.items(tick=3)

        self.cpu_reader.reset()

        cpu_utilization_dict = self.cpu_reader.items(tick=0)
        expected = 4
        self.assertEqual(expected, len(cpu_utilization_dict))

        cpu_utilization_dict = self.cpu_reader.items(tick=1)
        expected = 13
        self.assertEqual(expected, len(cpu_utilization_dict))

        cpu_utilization_dict = self.cpu_reader.items(tick=2)
        expected = 8
        self.assertEqual(expected, len(cpu_utilization_dict))

    def test_start_tick_not_in_first_file(self):
        self.cpu_reader = CpuReader(self.data_path, 2)

        cpu_utilization_dict = self.cpu_reader.items(tick=2)
        expected = 8
        self.assertEqual(expected, len(cpu_utilization_dict))

        cpu_utilization_dict = self.cpu_reader.items(tick=3)
        expected = 7
        self.assertEqual(expected, len(cpu_utilization_dict))


if __name__ == "__main__":
    unittest.main()
    print("hi")
