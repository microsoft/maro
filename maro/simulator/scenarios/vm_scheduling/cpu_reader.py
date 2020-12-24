# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.data_lib.binary_reader import BinaryReader


class CpuReader:
    """A wrapper class for the BinaryReader."""
    def __init__(self, data_path: str, start_tick: int):
        # Use for re-initialization.
        self._data_path = data_path
        self._cpu_reader = BinaryReader(self._data_path)
        self._cpu_item_picker = self._cpu_reader.items_tick_picker(
            start_time_offset=self._cpu_reader.header.starttime,
            end_time_offset=self._cpu_reader.header.endtime,
            time_unit="s"
        )
        while start_tick > self._cpu_reader.header.endtime:
            self._switch()

        self._init_data_path = self._data_path

    def _switch_to_next_file_name(self, data_path) -> str:
        """Switch to next file name."""
        file_name = data_path.split("-")
        file_name[2] = str(int(file_name[2]) + 1)
        new_data_path = "-".join(file_name)

        return new_data_path

    def _switch(self):
        """Switch to a new binary reader."""
        self._data_path = self._switch_to_next_file_name(self._data_path)
        self._cpu_reader = BinaryReader(self._data_path)
        self._cpu_item_picker = self._cpu_reader.items_tick_picker(
            start_time_offset=0,
            end_time_offset=self._cpu_reader.header.endtime - self._cpu_reader.header.starttime,
            time_unit="s"
        )

    def _pick_up_items(self, cur_items: dict, tick: int):
        end_time = 0
        for cpu in self._cpu_item_picker.items(tick=tick - self._cpu_reader.header.starttime):
            cur_items[cpu.vm_id] = cpu.cpu_utilization
            end_time = cpu.timestamp

        return cur_items, end_time

    def items(self, tick: int):
        cur_items = {}
        cur_items, end_time = self._pick_up_items(cur_items, tick)
        # The most end tick is 8638.
        if end_time == 8638:
            return cur_items
        # If the current tick is the end tick of the file, then switch to next file.
        while end_time == self._cpu_reader.header.endtime:
            new_file = os.path.expanduser(self._switch_to_next_file_name(self._data_path))
            if not os.path.exists(new_file):
                break
            self._switch()
            # Check the start tick of the new file is same as the end tick of the last file.
            if self._cpu_reader.header.starttime == end_time:
                cur_items, _ = self._pick_up_items(cur_items, tick)

        return cur_items

    def reset(self):
        self._data_path = self._init_data_path
        self._cpu_reader = BinaryReader(self._data_path)
        self._cpu_item_picker = self._cpu_reader.items_tick_picker(
            start_time_offset=self._cpu_reader.header.starttime,
            end_time_offset=self._cpu_reader.header.endtime,
            time_unit="s"
        )
