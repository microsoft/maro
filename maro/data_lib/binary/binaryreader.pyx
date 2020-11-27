# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

from collections import namedtuple
from maro.data_lib.binary.common cimport UCHAR, USHORT, ULONGLONG, LONGLONG, UINT, Meta, Field, BinHeader
from libc.stdint cimport int32_t, uint32_t


cdef class ItemContainerAccessor:
    cdef:
        ItemContainer* item
        int offset

    cdef set_item(self, ItemContainer* item):
        self.item = item

    cdef set_offset(self, int offset):
        self.offset = offset

    def get(self):
        pass

cdef class CharAccessor(ItemContainerAccessor):
    def get(self):
        return self.item.get[char](self.offset)

cdef class UCharAccessor(ItemContainerAccessor):
    def get(self):
        return self.item.get[UCHAR](self.offset)

cdef class ShortAccessor(ItemContainerAccessor):
    def get(self):
        return self.item.get[short](self.offset)

cdef class UShortAccessor(ItemContainerAccessor):
    def get(self):
        return self.item.get[USHORT](self.offset)

cdef class IntAccessor(ItemContainerAccessor):
    def get(self):
        return self.item.get[int32_t](self.offset)

cdef class UIntAccessor(ItemContainerAccessor):
    def get(self):
        return self.item.get[uint32_t](self.offset)

cdef class FloatAccessor(ItemContainerAccessor):
    def get(self):
        return self.item.get[float](self.offset)

cdef class DoubleAccessor(ItemContainerAccessor):
    def get(self):
        return self.item.get[double](self.offset)

cdef class LongAccessor(ItemContainerAccessor):
    def get(self):
        return self.item.get[LONGLONG](self.offset)

cdef class ULONGAcessor(ItemContainerAccessor):
    def get(self):
        return self.item.get[ULONGLONG](self.offset)


cdef dict field_access_map = {
    1: CharAccessor,
    2: UCharAccessor,
    3: ShortAccessor,
    4: UShortAccessor,
    5: IntAccessor,
    6: UIntAccessor,
    7: LongAccessor,
    8: ULONGAcessor,
    9: FloatAccessor,
    10: DoubleAccessor,
    11: ULONGAcessor,
}


cdef unit_seconds(str unit):
    # default for second
    cdef UINT seconds = 1

    if unit == "m":
        seconds = 60
    elif unit == "h":
        seconds = 60 * 60
    elif unit == "d":
        seconds = 60 * 60 * 24

    return seconds


cdef calc_time_offset(ULONGLONG start_time, UINT offset, str unit):
    """Calculate time by offset and time unit."""
    cdef UINT seconds_per_unit = unit_seconds(unit)

    return offset * seconds_per_unit + start_time


cdef class ItemTickPicker:
    """Wrapper to support get items by tick."""
    cdef:
        object _item_generaotr
        ULONGLONG _starttime
        str _time_unit
        object _cached_item

    def __init__(self, item_generaotr, starttime: int, time_unit: str):
        self._item_generaotr = item_generaotr
        self._starttime = starttime
        self._time_unit = time_unit
        self._cached_item = None

    def items(self, tick: int):
        """Get items for specified ticks.

        NOTE:
            This method will compare timestamp of item to pick, so if the timestamp is not equal to current tick
            the it will be dropped.

        Args:
            tick (int): Tick of items to filter.

        Returns:
            Generator: Generator to iterate items of current tick.
        """
        cdef UINT _tick = tick
        cdef UINT seconds_per_unit = unit_seconds(self._time_unit)
        cdef ULONGLONG ticks_in_seconds = self._starttime + _tick * seconds_per_unit

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


cdef class MaroBinaryReader:
    def __init__(self):
        self._item_nt = None

    @property
    def start_timestamp(self):
        """int: Start timestamp."""
        return self._header.start_timestamp

    @property
    def end_timestamp(self):
        """int: End timestamp"""
        return self._header.end_timestamp

    @property
    def item_count(self):
        """int: Total items in binary file."""
        return self._header.total_items

    @property
    def item_size(self):
        """int: Size of each item (in byte)."""
        return self._header.item_size

    @property
    def file_type(self):
        """int: Type of binary file (1 for now)."""
        return self._header.file_type

    @property
    def file_version(self):
        """int: Version of this file (customized, default is 0)."""
        return self._header.file_version

    @property
    def converter_version(self):
        """int: Version of the converter that converted the binary file."""
        return self._header.converter_version
        

    def open(self, file: str):
        """Open specified binary file.

        Args:
            file (str): Path to the binary file to read.
        """
        self._reader.open(file.encode())

        # Construct item namedtuple
        cdef const Meta* meta = self._reader.get_meta()

        cdef ItemContainerAccessor acc

        cdef int i = 0

        self._header = self._reader.get_header()

        field_names = []

        self._item_fields_accessor = []

        for i in range(meta.fields.size()):
            field_names.append(meta.fields[i].alias.decode())

            acc = field_access_map[meta.fields[i].type]()

            acc.set_offset(meta.fields[i].start_index)

            self._item_fields_accessor.append(acc)

        self._item_nt = namedtuple("BinaryItem", field_names)

    def close(self):
        """Close the file."""
        self._reader.close()

    def reset(self):
        """Reset internal state, usually used to read from beginning again."""
        self._reader.reset()

    def items_tick_picker(self, start_time_offset: int = 0, end_time_offset: int = None, time_unit: str = "s"):
        """Get a picker that used to get items by tick within specified time range.

        Args:
            start_time_offset (int): Start of the range, offset to start timestamp in binary file.
            end_time_offset (int): End of the range, offset to start timestamp in binary file.
            time_unit (str): Time unit to calculate the time offset, it support 's' for second, 'm' for minute, 'h' for hour
                and 'd' for day.
    
        Returns:
            ItemTickPicker: A picker object for easy accessing.
        """
        cdef ULONGLONG start_time = calc_time_offset(self._header.start_timestamp, start_time_offset, time_unit)

        cdef ULONGLONG end_time = INVALID_FILTER

        if end_time_offset is not None:
            end_Time = calc_time_offset(self._header.start_timestamp, end_time_offset, time_unit)

        self._reader.set_filter(start_time, end_time)

        return ItemTickPicker(self.items(), self._header.start_timestamp, time_unit)

    def set_filter(self, start: int, end: int = None):
        """Enable and set the filter range.

        NOTE:
            This will affect the result of 'items' method, you can disable it by call 'disable_filter'.

        Args:
            start (int): Start timestamp of the range (in UTC).
            end (int): End timestamp of the range (in UTC).
        """
        if end == None:
            end = INVALID_FILTER

        self._reader.set_filter(start, end)

    def disable_filter(self):
        """Disable current filter, this will may the items iterate from beginning."""
        self._reader.disable_filter()

    def items(self):
        """Iterate items from binary file, apply filters if enabled.

        NOTE:
            This method need to be reset if you need to use it again, or it cannnot reset its internal states.

        Returns:
            Generator: Generator to iterate items in binary file.
        """
        cdef ItemContainer* item
        cdef ItemContainerAccessor acc
        cdef list values = []

        item = self._reader.next_item()

        while item:
            values.clear()

            for acc in self._item_fields_accessor:
                acc.set_item(item)
                values.append(acc.get())

            yield self._item_nt._make(values)

            item = self._reader.next_item()

        return None
