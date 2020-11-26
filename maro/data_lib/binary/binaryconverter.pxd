# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++


cimport cython

from cpython cimport bool
from libcpp.string cimport string
from maro.data_lib.binary.common cimport UCHAR, ULONGLONG, LONGLONG, UINT, Meta, Field, BinHeader
from libc.stdint cimport int32_t


cdef extern from "cpp/binarywriter.cpp":
    pass


cdef extern from "cpp/binarywriter.h" namespace "maro::datalib":
    cdef cppclass BinaryWriter:
        BinaryWriter()

        void open(string output_folder, string file_name, string file_type, int32_t file_version)

        void load_meta(string meta_file)

        void add_csv(string csv_file)

        void set_start_timestamp(ULONGLONG start_timestamp)

cdef class MaroBinaryConverter:
    cdef:
        BinaryWriter _writer
