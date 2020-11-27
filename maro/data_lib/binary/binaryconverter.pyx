# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

cimport cython

from cpython cimport bool
from libcpp.string cimport string
from maro.data_lib.binary.common cimport UCHAR, ULONGLONG, LONGLONG, UINT, Meta, Field, BinHeader
from libc.stdint cimport int32_t


cdef class MaroBinaryConverter:
    def load_meta(self, meta_file: str):
        # TODO: convert yaml to toml if input is yaml
        self._writer.load_meta(meta_file.encode())

    def open(self, output_file: str, file_type: str = "NA", file_version:int = 0):
        cdef str _ft = file_type
        cdef int32_t _fv = file_version

        self._writer.open(output_file.encode(), _ft.encode(), _fv)

    def set_start_timestamp(self, start_timestamp:int):
        self._writer.set_start_timestamp(start_timestamp)

    def add_csv(self, csv_file:str):
        self._writer.add_csv(csv_file.encode())

    def close(self):
        self._writer.close()
