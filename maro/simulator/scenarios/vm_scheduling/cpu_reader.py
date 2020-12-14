# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.data_lib.binary_reader import BinaryReader


class CpuReader:

    def __init__(self, data_path: str):
        self._data_path = data_path
        self._cpu_reader = BinaryReader(self._data_path)
        self._cpu_item_picker = self._cpu_reader.items_tick_picker(
            start_time_offset=self._cpu_reader.header.starttime,
            end_time_offset=self._cpu_reader.header.endtime,
            time_unit="s"
        )
        self.count = 0

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

    def items(self, tick: int):
        cur_items = {}
        end_time = 0
        for cpu in self._cpu_item_picker.items(tick=tick - self._cpu_reader.header.starttime):
            cur_items[cpu.vm_id] = cpu.cpu_utilization
            end_time = cpu.timestamp
            self.count += 1
        # Switch to a new file at the end of the file.
        if self.count == self._cpu_reader.header.item_count:
            self._switch()
            self.count = 0
            # Check the start time in new file is equal to the end time of the previous file.
            if self._cpu_reader.header.starttime == end_time:
                for cpu in self._cpu_item_picker.items(tick=tick - self._cpu_reader.header.starttime):
                    cur_items[cpu.vm_id] = cpu.cpu_utilization
                    self.count += 1

        return cur_items
