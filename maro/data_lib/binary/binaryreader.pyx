# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

from collections import namedtuple
from maro.data_lib.binary.common cimport UCHAR, ULONGLONG, LONGLONG, UINT, Meta, Field, BinHeader
from libc.stdint cimport int32_t


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

cdef class ShortAccessor(ItemContainerAccessor):
    def get(self):
        return self.item.get[short](self.offset)

cdef class IntAccessor(ItemContainerAccessor):
    def get(self):
        return self.item.get[int32_t](self.offset)

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


field_access_map = {
    1: ShortAccessor,
    2: IntAccessor,
    3: LongAccessor,
    4: FloatAccessor,
    5: DoubleAccessor,
    6: ULONGAcessor
}


cdef class MaroBinaryReader:
    def __init__(self):
        self._item_nt = None

    @property
    def start_time(self):
        return self._header.start_timestamp

    @property
    def end_time(self):
        return self._header.end_timestamp

    @property
    def item_count(self):
        return self._header.total_items

    @property
    def file_type(self):
        return self._header.file_type

    @property
    def file_version(self):
        return self._header.file_version

    @property
    def converter_version(self):
        return self._header.converter_version
        

    def open(self, file: str):
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

    def reset(self):
        self._reader.reset()

    def set_filter(self, start: int, end: int = None):
        if end == None:
            end = INVALID_FILTER

        self._reader.set_filter(start, end)

    def items(self):
        """
        cdef ItemContainer* item = self._reader.next_item()
        cdef ItemContainerAccessor acc

        if item:
            values = []

            for acc in self._item_fields_accessor:
                acc.set_item(item)
                values.append(acc.get())

            return self._item_nt._make(values)
        else:
            return None
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
