# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

cimport cython

from cpython cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from libc.stdint cimport uint32_t


ctypedef unsigned char UCHAR
ctypedef unsigned short USHORT
ctypedef unsigned long long ULONGLONG
ctypedef long long LONGLONG
ctypedef uint32_t UINT


cdef extern from "cpp/common.cpp":
    pass


cdef extern from "cpp/common.h" namespace "maro::datalib":
    cdef cppclass BinHeader:
        unsigned char file_type
        char custom_file_type[3]
        char identifier[5]
        char utc_offset

        UINT converter_version
        UINT file_version
        UINT item_size
        UINT meta_size

        ULONGLONG total_items
        ULONGLONG start_timestamp
        ULONGLONG end_timestamp

        ULONGLONG reserved1
        ULONGLONG reserved2
        ULONGLONG reserved3
        ULONGLONG reserved4

    cdef cppclass Field:
        unsigned char type
        uint32_t size
        uint32_t start_index
        string column
        string alias

    cdef cppclass Meta:
        char utc_offset

        int size() const

        string get_alias(int field_index) const

        UCHAR get_type(int field_index) const

        uint32_t get_start_index(int field_index) const

cdef extern from "cpp/metaparser.cpp":
    pass

cdef extern from "cpp/metaparser.h" namespace "maro::datalib":
    pass