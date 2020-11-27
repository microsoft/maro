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
    cdef:
        BinaryReader _reader

        object _item_nt

        list _item_fields_accessor

        const BinHeader* _header