# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

cimport cython

from cpython cimport bool
from libcpp.string cimport string
from collections import namedtuple
from maro.data_lib.binary.common cimport UCHAR, ULONGLONG, LONGLONG, UINT, Meta, Field, BinHeader


cdef ULONGLONG INVALID_FILTER = 0


cdef extern from "cpp/itemcontainer.cpp":
    pass


cdef extern from "cpp/itemcontainer.h" namespace "maro::datalib":

    cdef cppclass ItemContainer:
        T get[T](int offset);


cdef extern from "cpp/binaryreader.cpp":
    pass


cdef extern from "cpp/binaryreader.h" namespace "maro::datalib":
    cdef cppclass BinaryReader:
        void open(string bin_file) except +
        void close() except +

        ItemContainer *next_item() except +

        const Meta *get_meta() except +
        const BinHeader *get_header() except +

        void reset()

        void set_filter(ULONGLONG start, ULONGLONG end) except +
        void disable_filter()


cdef class MaroBinaryReader:
    """Reader that used to read binary files that generated with MaroBinaryConverter.

    NOTE:
        Method 'items' used to iterate items from binary file, if you need to use it again to read from beginning, then you need
        to call reset first, or will not result.

        Different with items, method 'items_tick_picker' used to filter items in specified range, and return items by tick, it will
        call reset internally, so do not need to reset, but do need to call the method again.

        The converter version should be same with reader version, or it will cause exception.

    .. code-block:: python

        reader = Reader()

        # Open target binary file.
        reader.open("path/to/my/file.bin")

        # Iterate all items
        for item in reader.items():
            # do something

        # Reset before re-use
        reader.reset()

        # Again
        for item in reader.items():
            # do something

        # Or we can use filter, then pick by tick
        picker = reader.items_tick_picker(0, 10, time_unit="m")

        # Pick by tick
        for tick in range(11):
            for item in picker.items(tick):
                # do something

        # Again, but no reset
        picker = reader.items_tick_picker(0, 10, time_unit="m")
    """
    cdef:
        BinaryReader _reader

        object _item_nt

        list _item_fields_accessor

        const BinHeader* _header
