# Copyright (c) Microsoft Corporation.
# Licensed under the MIT licence

import os
import tempfile
import unittest
from dateutil.tz import gettz
from maro.data_lib import MaroBinaryConverter, MaroBinaryReader
from maro.simulator.scenarios.helpers import utc_timestamp_to_timezone

ny_tz = gettz("America/New_York")


class TestBinaryConverter(unittest.TestCase):
    def test_convert_with_events(self):
        out_dir = tempfile.mkdtemp()

        out_bin = os.path.join(out_dir, "trips.bin")

        meta_file = os.path.join(
            "tests", "data", "data_lib", "case_1", "meta.toml")
        csv_file = os.path.join("tests", "data", "data_lib", "trips.csv")

        bct = MaroBinaryConverter()
        bct.open(out_bin)
        bct.load_meta(meta_file)

        # add and convert 1st csv file
        bct.add_csv(csv_file)

        # add again will append to the end ignore the order
        bct.add_csv(csv_file)

        # flush will close the file, cannot add again
        bct.close()

        # check if output exist
        self.assertTrue(os.path.exists(out_bin))

        # check content
        reader = MaroBinaryReader()
        reader.open(out_bin)

        # start tick should be smallest one
        start_date = utc_timestamp_to_timezone(reader.start_timestamp, ny_tz)

        self.assertEqual(start_date.year, 2019)
        self.assertEqual(start_date.month, 1)
        self.assertEqual(start_date.day, 1)
        self.assertEqual(start_date.hour, 0)
        self.assertEqual(start_date.minute, 0)
        self.assertEqual(start_date.second, 0)

        end_date = utc_timestamp_to_timezone(reader.end_timestamp, ny_tz)

        self.assertEqual(end_date.year, 2019)
        self.assertEqual(end_date.month, 1)
        self.assertEqual(end_date.day, 1)
        self.assertEqual(end_date.hour, 0)
        self.assertEqual(end_date.minute, 5)
        self.assertEqual(end_date.second, 0)

        # there should be double items as trips.csv
        self.assertEqual(4*2, reader.item_count)

        # 20 byte
        self.assertEqual(14, reader.item_size)

        start_station_index = [0, 0, 1, 0]

        idx = 0

        # check iterating interface
        for item in reader.items():
            # check if fields same as meta
            self.assertTupleEqual(
                ('timestamp', 'durations', 'src_station', 'dest_station'), item._fields)

            # check item start station index
            self.assertEqual(start_station_index[idx % len(
                start_station_index)], item.src_station)

            idx += 1

        # check if filter works as expected
        picker = reader.items_tick_picker(
            start_time_offset=0,  end_time_offset=1, time_unit="m")

        l = len([item for item in picker.items(0)])

        # although there are 2 items that match the condition, but they not sorted, reader will not try to read to the end, but
        # to the first item which not match the condition
        self.assertEqual(1, l)

        print(reader.start_timestamp)

        picker = reader.items_tick_picker(start_time_offset=1, time_unit='m')

        l = 0

        for tick in range(10):
            for item in picker.items(tick):
                l += 1

        # ItemsItckPicker will filter items that not match current tick
        # reader.items() will not
        self.assertEqual(4, l)

    def test_convert_with_starttimestamp(self):
        out_dir = tempfile.mkdtemp()

        out_bin = os.path.join(out_dir, "trips.bin")

        meta_file = os.path.join(
            "tests", "data", "data_lib", "case_2", "meta.toml")
        csv_file = os.path.join("tests", "data", "data_lib", "trips.csv")

        bct = MaroBinaryConverter()
        bct.open(out_bin)
        bct.load_meta(meta_file)

        # 12/31/2018 @ 11:59pm (NY) in UTC
        bct.set_start_timestamp(1546318800 - 60)

        bct.add_csv(csv_file)

        # flush will close the file, cannot add again
        bct.close()

        reader = MaroBinaryReader()
        reader.open(out_bin)

        # check header
        self.assertEqual(1546318800 - 60, reader.start_timestamp)

        # then tick 0 will not be 2019/01/01 00:00:00
        picker = reader.items_tick_picker(end_time_offset=1, time_unit='m')

        l = len([item for item in picker.items(0)])

        self.assertEqual(0, l)

        # # it should be tick 1 for now
        l = len([item for item in picker.items(1)])

        self.assertEqual(1, l)

    def test_convert_without_meta(self):
        out_dir = tempfile.mkdtemp()

        out_bin = os.path.join(out_dir, "trips.bin")

        csv_file = os.path.join("tests", "data", "data_lib", "trips.csv")

        # 12/31/2018 @ 11:59pm (UTC)
        with self.assertRaises(RuntimeError) as ctx:
            bct = MaroBinaryConverter()
            bct.open(out_bin)

            bct.add_csv(csv_file)

    def test_convert_without_open(self):
        csv_file = os.path.join("tests", "data", "data_lib", "trips.csv")

        # 12/31/2018 @ 11:59pm (UTC)
        with self.assertRaises(RuntimeError) as ctx:
            bct = MaroBinaryConverter()

            bct.add_csv(csv_file)


if __name__ == "__main__":
    unittest.main()
