# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import timeit
import unittest

from pprint import pprint

from maro.cli.data_pipeline.data_process import generate, list_env
from maro.event_buffer import EventBuffer
from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import CpuReader, PlaceAction, VmSchedulingBusinessEngine

from tests.utils import next_step


class CpuReaderTest(unittest.TestCase):
    def setUp(self):
        data_path = "/mnt/d/kuanwei/data/test_10k/build/vm_cpu_readings-file-1-of-195.bin"
        self.cpu_reader = CpuReader(data_path, 0)

    def tearDown(self):
        self.cpu_reader.reset()

    def test_first_file_first_tick(self):
        cpu_utilization_dict = self.cpu_reader.items(tick=0)
        expected = 833
        self.assertEqual(expected, len(cpu_utilization_dict))

    def test_first_file_last_tick(self):
        cpu_utilization_dict = self.cpu_reader.items(tick=44)
        expected = 833
        self.assertEqual(expected, len(cpu_utilization_dict))

    def test_switch_file(self):
        cpu_utilization_dict = self.cpu_reader.items(tick=44)
        cpu_utilization_dict = self.cpu_reader.items(tick=45)
        expected = 831
        self.assertEqual(expected, len(cpu_utilization_dict))

    def test_last_file(self):
        cpu_utilization_dict = {}
        for i in range(8627):
            cpu_utilization_dict = self.cpu_reader.items(tick=i)

        expected = 816
        self.assertEqual(expected, len(cpu_utilization_dict))

        cpu_utilization_dict = self.cpu_reader.items(tick=8638)
        expected = 819
        self.assertEqual(expected, len(cpu_utilization_dict))


class BusinessEngineTest(unittest.TestCase):
    def setUp(self):
        self.event_buffer = EventBuffer()
        self.business_engine = VmSchedulingBusinessEngine(
            event_buffer=self.event_buffer,
            topology="azure.2019.10k",
            start_tick=0,
            max_tick=10,
            snapshot_resolution=1,
            max_snapshots=None,
            additional_options={}
        )

    def test_first_tick(self):
        next_step(self.event_buffer, self.business_engine, 0)
        pm_amount = len(self.business_engine.frame.pms)

        self.assertEqual(100, pm_amount)

        pm_0 = self.business_engine.frame.pms[0]

        self.assertEqual(32, pm_0.cpu_cores_capacity)
        self.assertEqual(128, pm_0.memory_capacity)
        self.assertEqual(0, pm_0.cpu_cores_allocated)
        self.assertEqual(0, pm_0.memory_allocated)


if __name__ == "__main__":
    unittest.main()
