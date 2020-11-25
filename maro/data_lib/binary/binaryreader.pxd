# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

cimport cython

from cpython cimport bool
from libcpp.string cimport string
from collections import namedtuple
from maro.data_lib.binary.common cimport UCHAR, ULONGLONG, LONGLONG, UINT, Meta, Field, BinHeader

cdef extern from "cpp/itemcontainer.cpp":
    pass

cdef extern from "cpp/itemcontainer.h" namespace "maro::datalib":

    cdef cppclass ItemContainer:
        void set_buffer(char *buffer);

        void set_offset(UINT offset);

        T get[T](int offset);


cdef extern from "cpp/binaryreader.cpp":
    pass


cdef extern from "cpp/binaryreader.h" namespace "maro::datalib":
    cdef cppclass BinaryReaderIterator:
        BinaryReaderIterator &operator++()
        ItemContainer *operator*()
        bool operator!=(const BinaryReaderIterator &bri)

    cdef cppclass BinaryReader:
        void open(string bin_file)

        ItemContainer *next_item()

        const Meta *get_meta()

        BinaryReaderIterator begin()

        BinaryReaderIterator end()

        void reset()


cdef class MaroBinaryReader:
    cdef:
        BinaryReader _reader

        object _item_nt

        list _item_fields_accessor