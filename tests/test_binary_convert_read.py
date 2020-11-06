# Copyright (c) Microsoft Corporation.
# Licensed under the MIT licence

import os
import tempfile
import unittest

from maro.data_lib import BinaryConverter, BinaryReader
from maro.data_lib.item_meta import BinaryMeta


class TestBinaryConverter(unittest.TestCase):
    def test_convert_with_events(self):
        out_dir = tempfile.mkdtemp()

        out_bin = os.path.join(out_dir, "trips.bin")

        meta_file = os.path.join("tests", "data", "data_lib", "case_1", "meta.yml")
        csv_file = os.path.join("tests", "data", "data_lib", "trips.csv")

        bct = BinaryConverter(out_bin, meta_file)

        # add and convert 1st csv file
        bct.add_csv(csv_file)

        # add again will append to the end ignore the order
        bct.add_csv(csv_file)

        # flush will close the file, cannot add again
        bct.flush()


        # check if output exist
        self.assertTrue(os.path.exists(out_bin))

        # check content
        reader = BinaryReader(out_bin)

        # start tick should be smallest one
        start_date = reader.start_datetime

        self.assertEqual(start_date.year, 2019)
        self.assertEqual(start_date.month, 1)
        self.assertEqual(start_date.day, 1)
        self.assertEqual(start_date.hour, 0)
        self.assertEqual(start_date.minute, 0)
        self.assertEqual(start_date.second, 0)

        end_date = reader.end_datetime

        self.assertEqual(end_date.year, 2019)
        self.assertEqual(end_date.month, 1)
        self.assertEqual(end_date.day, 1)
        self.assertEqual(end_date.hour, 0)
        self.assertEqual(end_date.minute, 5)
        self.assertEqual(end_date.second, 0)     


        # there should be double items as trips.csv
        self.assertEqual(4*2, reader.header.item_count)

        # 20 byte
        self.assertEqual(20, reader.header.item_size)   
        
        start_station_index = [0, 0, 1, 0]

        idx = 0

        # check iterating interface
        for item in reader.items():
            # check if fields same as meta
            self.assertTupleEqual(('timestamp', 'durations', 'src_station', 'dest_station'), item._fields)

            # check item start station index
            self.assertEqual(start_station_index[idx % len(start_station_index)], item.src_station)

            idx += 1
        
        # check if filter works as expected
        l = len([item for item in reader.items(end_time_offset=0, time_unit="m")])

        # although there are 2 items that match the condition, but they not sorted, reader will not try to read to the end, but 
        # to the first item which not match the condition
        self.assertEqual(1, l)

        l = len([item for item in reader.items(start_time_offset=1, time_unit='m')])

        # reader will try to read 1st one that > end tick, so there should be 6 items 
        self.assertEqual(6, l)

    def test_convert_without_events(self):
        out_dir = tempfile.mkdtemp()

        out_bin = os.path.join(out_dir, "trips.bin")

        meta_file = os.path.join("tests", "data", "data_lib", "case_2", "meta.yml")
        csv_file = os.path.join("tests", "data", "data_lib", "trips.csv")

        bct = BinaryConverter(out_bin, meta_file)

        bct.add_csv(csv_file)

        # flush will close the file, cannot add again
        bct.flush()

        reader = BinaryReader(out_bin)

        meta: BinaryMeta = reader.meta

        self.assertIsNotNone(meta)

        # check events
        self.assertListEqual(["require_bike", "return_bike", "rebalance_bike", "deliver_bike"], [event.display_name for event in meta.events])

        self.assertListEqual(["RequireBike", "ReturnBike", "RebalanceBike", "DeliverBike"], [event.type_name for event in meta.events])

        self.assertEqual("RequireBike", meta.default_event_name)
        self.assertIsNone(meta.event_attr_name)

    def test_convert_with_starttimestamp(self):
        out_dir = tempfile.mkdtemp()

        out_bin = os.path.join(out_dir, "trips.bin")

        meta_file = os.path.join("tests", "data", "data_lib", "case_2", "meta.yml")
        csv_file = os.path.join("tests", "data", "data_lib", "trips.csv")

        #12/31/2018 @ 11:59pm (UTC)
        bct = BinaryConverter(out_bin, meta_file, utc_start_timestamp=1546300740)

        bct.add_csv(csv_file)

        # flush will close the file, cannot add again
        bct.flush()

        reader = BinaryReader(out_bin)

        # check header
        self.assertEqual(1546300740, reader.header.starttime)

        # then tick 0 will not be 2019/01/01 00:00:00
        l = len([item for item in reader.items(end_time_offset=0, time_unit='m')])

        self.assertEqual(0, l)

        # it should be tick 1 for now
        l = len([item for item in reader.items(end_time_offset=1, time_unit='m')])

        self.assertEqual(1, l)


    def test_convert_without_meta_timestamp(self):
        out_dir = tempfile.mkdtemp()

        out_bin = os.path.join(out_dir, "trips.bin")

        meta_file = os.path.join("tests", "data", "data_lib", "case_3", "meta.yml")
        csv_file = os.path.join("tests", "data", "data_lib", "trips.csv")

        #12/31/2018 @ 11:59pm (UTC)
        with self.assertRaises(Exception) as ctx:
            bct = BinaryConverter(out_bin, meta_file)


if __name__ == "__main__":
    unittest.main()
