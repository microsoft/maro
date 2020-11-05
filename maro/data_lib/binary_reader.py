# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import mmap
import os
import sys
import warnings
from datetime import datetime
from typing import Union

from dateutil.relativedelta import relativedelta
from dateutil.tz import UTC

from maro.data_lib.common import VERSION, FileHeader, header_struct
from maro.data_lib.item_meta import BinaryMeta

# used to get correct datetime with negative timestamp on Windows
timestamp_start = datetime(1970, 1, 1, 0, 0, 0, tzinfo=UTC)


def unit_seconds(unit: str):
    # default for second
    seconds = 1

    if unit == "m":
        seconds = 60
    elif unit == "h":
        seconds = 60 * 60
    elif unit == "d":
        seconds = 60 * 60 * 24

    return seconds


def calc_time_offset(start_time: int, offset: int, unit: str):
    """Calculate time by offset and time unit."""
    seconds_per_unit = unit_seconds(unit)

    return offset * seconds_per_unit + start_time


class ItemBuffer:
    """In-memory buffer for binary data."""

    def __init__(self, number_of_item: int, meta: BinaryMeta, enable_adjust_ratio: bool = False):
        self._meta = meta
        self._enable_adjust_ratio = enable_adjust_ratio
        self._bytes = memoryview(bytearray(number_of_item * meta.item_size))
        # valid items in buffer
        self.item_number = 0

    def items(self):
        index = 0

        while index < self.item_number:
            yield self._meta.item_from_bytes(self._bytes[index * self._meta.item_size:], self._enable_adjust_ratio)

            index += 1

    def write(self, contents: Union[bytes, bytearray, memoryview]):
        if contents is None:
            self.item_number = 0

            return

        self._bytes[0:len(contents)] = contents

        self.item_number = int(len(contents) / self._meta.item_size)


class ItemTickPicker:
    """Wrapper to support get items by tick."""

    def __init__(self, item_generaotr, starttime: int, time_unit: str):
        self._item_generaotr = item_generaotr
        self._starttime = starttime
        self._time_unit = time_unit
        self._cached_item = None

    def items(self, tick: int):
        """Get items for specified ticks.

        NOTE:
            This method will compare timestamp of item to pick.
        """
        seconds_per_unit = unit_seconds(self._time_unit)
        ticks_in_seconds = self._starttime + tick * seconds_per_unit

        while True:
            item = self._cached_item
            # clear the cache
            self._cached_item = None

            if item is None:
                try:
                    item = next(self._item_generaotr)
                except StopIteration:
                    break

            if item.timestamp >= ticks_in_seconds:
                delta = item.timestamp - ticks_in_seconds

                # with in one unit
                if int(delta) < seconds_per_unit:
                    yield item
                else:
                    # or keep it for next time
                    self._cached_item = item
                    break
            else:
                # here we can log items that not sorted
                pass


class BinaryReader:
    """Read binary file converted by csv converter.

    Examples:

        .. code-block:: python

            reader = BinaryReader(bin_file)

            # Read items in between 0-10 minute (relative to binary start time).
            for item in reader.items(0, 10, time_unit="m"):
                print(item)

            # Or get a picker that support query by tick sequentially.
            picker = reader.items_tick_picker(0, 10, time_unit="m"):

            for tick in range(0, 10):
                for item in picker.items(tick):
                    print(item)

    Args:
        file_path(str): Binary file path to read.
        enable_value_adjust(bool): If reader should adjust the value of fields that enabled
            'value_adjust' feature in meta randomly.
        buffer_size(int): Size of in-memory buffer.
    """

    def __init__(self, file_path: str, enable_value_adjust: bool = False, buffer_size: int = 100):
        self._enable_value_adjust = enable_value_adjust

        self.header: FileHeader = None
        self._meta = BinaryMeta()

        self._buffer_size = buffer_size
        self._file_fp = None
        self._mmap: mmap.mmap = None
        if file_path.startswith("~"):
            file_path = os.path.expanduser(file_path)
        self._file_fp = open(file_path, "rb")

        if sys.platform == "win32":
            self._mmap = mmap.mmap(
                self._file_fp.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            self._mmap = mmap.mmap(
                self._file_fp.fileno(), 0, prot=mmap.PROT_READ)

        self._read_header()
        self._read_meta()

        # double buffer to read data
        self._item_buffer = ItemBuffer(
            buffer_size, self._meta, enable_value_adjust)

        # contains starttime offset related file offset, used in items() method
        # use this to speedup the querying
        self._starttime_offset_history = {}

        # move the pointer to data area
        self._mmap.seek(self.header.data_offset)

        # data length (in byte) we already loaded, used to check data boundary
        self._readed_data_size = 0

    @property
    def meta(self) -> BinaryMeta:
        """BinaryMeta: Meta data in binary file."""
        return self._meta

    @property
    def start_datetime(self) -> datetime:
        """datetime: Start datetime of this file (UTC)."""
        return self._to_utc_datetime(self.header.starttime)

    @property
    def end_datetime(self) -> datetime:
        """datetime: End datetime of this file (UTC)."""
        return self._to_utc_datetime(self.header.endtime)

    def items_tick_picker(self, start_time_offset: int = 0, end_time_offset: int = None, time_unit: str = "s"):
        """Filter items by specified time range, and then pick by tick sequentially.

        Args:
            start_time_offset(int): Specified the which tick (in seconds) to start.
            end_time_offset(int): Specified the end tick (in seconds) to start.
            time_unit (str): Unit of time used to calculate offset, 's': seconds, 'm': minute, 'h': hour, 'd': day.

        Returns:
            ItemTickPicker: A picker object that support get items by tick in specified range.
        """
        item_filter = self.items(start_time_offset, end_time_offset, time_unit)

        return ItemTickPicker(item_filter, self.header.starttime, time_unit)

    def items(self, start_time_offset: int = 0, end_time_offset: int = None, time_unit: str = "s"):
        """Get all items in specified time range.

        Args:
            start_time_offset(int): Specified the which tick (in seconds) to start.
            end_time_offset(int): Specified the end tick (in seconds) to start.
            time_unit (str): Unit of time used to calculate offset, 's': seconds, 'm': minute, 'h': hour, 'd': day.

        Returns
            interable: Items in specified range.
        """
        # reset to read from beginning
        self.reset()

        # default offset
        offset = self.header.data_offset

        # time range to filter
        start_time = calc_time_offset(self.header.starttime, start_time_offset, time_unit)

        if end_time_offset is None:
            end_time = self.header.endtime
        else:
            end_time = calc_time_offset(
                self.header.starttime, end_time_offset, time_unit)

        # check if we have used this filter
        has_filter_history = False

        if start_time_offset in self._starttime_offset_history:
            has_filter_history = True

            offset = self._starttime_offset_history[start_time_offset]

        # fulfill buffer for first time using
        # seek to the data part to go through all the items
        self._mmap.seek(offset)

        self._fulfill_buffer()

        pre_mmap_offset = self._mmap.tell()

        while True:
            # read and return an item from buffer
            buffer = self._item_buffer

            if buffer.item_number == 0:
                break

            is_finished = False

            for item in buffer.items():
                if start_time <= item.timestamp <= end_time:
                    # record the filter history
                    if not has_filter_history:
                        has_filter_history = True

                        # return to the start of the buffer
                        pos = pre_mmap_offset - buffer.item_number * self._meta.item_size

                        self._starttime_offset_history[start_time_offset] = pos

                    yield item

                if item.timestamp > end_time:
                    is_finished = True
                    return

            if not is_finished:
                # then start another one
                pre_mmap_offset = self._mmap.tell()

                self._fulfill_buffer()
            else:
                break

    def reset(self):
        """Reset binary reader."""
        self._readed_data_size = 0

    def __del__(self):
        """Clear resources."""
        self.close()

    def close(self):
        """Close file."""
        if self._mmap and not self._mmap.closed:
            self._mmap.close()

            self._mmap = None

        if self._file_fp and not self._file_fp.closed:
            self._file_fp.close()

            self._file_fp = None

    def _to_utc_datetime(self, timestamp: int):
        """Convert timestamp into datetime."""

        # TODO: make it as a common method
        if sys.platform == "win32":
            return (timestamp_start + relativedelta(seconds=timestamp))
        else:
            return datetime.utcfromtimestamp(timestamp).replace(tzinfo=UTC)

    def _read_header(self):
        """Read header part."""
        header_bytes = memoryview(self._mmap[0:header_struct.size])

        self.header = FileHeader._make(header_struct.unpack_from(header_bytes))

        # validate header
        # if current version less than file, then a warning
        if VERSION < self.header.version:
            warnings.warn(
                "File version is greater than current reader version, may cause unknown behavior!.")

    def _read_meta(self):
        """Read meta part."""
        meta_bytes = self._mmap[self.header.meta_offset:
                                self.header.meta_offset + self.header.meta_size]

        self._meta.from_bytes(meta_bytes)

    def _fulfill_buffer(self):
        """Fulfill buffer from file."""

        buffer = self._item_buffer

        size_to_read = self._meta.item_size * self._buffer_size
        remaining_size = self.header.data_size - self._readed_data_size

        size_to_read = min(size_to_read, remaining_size)

        if size_to_read <= 0:
            buffer.write(None)
        else:
            item_bytes = self._mmap.read(size_to_read)

            self._readed_data_size += len(item_bytes)

            buffer.write(item_bytes)


__all__ = ['BinaryReader']
