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
        """Load meta from specified file to prepare for further converting.

        Args:
            meta_file (str): Path to meta file.
        """
        self._writer.load_meta(meta_file.encode())

    def open(self, output_file: str, file_type: str = "NA", file_version:int = 0):
        """Open/create output file to hold binary result.

        Args:
            output_file (str): Path to output file.
            file_type (str): Customized file type, length must be 2.
            file_version (int): Customized file versoin.
        """
        if len(file_type) != 2:
            raise Exception("Length of customized file type must be 2.")

        self._writer.open(output_file.encode(), file_type.encode(), file_version)

    def set_start_timestamp(self, start_timestamp:int):
        """Set start timestamp in binary file.

        Args:
            start_timestamp (int): Start timestamp need to set, should be an UTC timestamp.
        """
        self._writer.set_start_timestamp(start_timestamp)

    def add_csv(self, csv_file:str):
        """Add a CSV file to convert.

        NOTE:
            This method should be called after open and load_meta.

        Args:
            csv_file (str): Path to CSV file to convert.
        """
        self._writer.add_csv(csv_file.encode())

    def close(self):
        """Close current file, this will stop furthure converting."""
        self._writer.close()
