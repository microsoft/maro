# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++


cimport cython

from libcpp.string cimport string


cdef extern from "cpp/binarycombine.cpp":
    pass


cdef extern from "cpp/binarycombine.h" namespace "maro::datalib":
    cdef cppclass BinaryCombine:
        BinaryCombine()
        void open(string output_file)
        void combine(string bin_file1, string bin_file2)
        void close()


cdef class MaroBinaryCombine:
    """Util used to combine binary files, group them by timestamp.

    NOTE:
        This class will not sort binary files, just pick items from head to tail, and compare which
        one is small then append to the output file.

        .. code-block:: python

            combine = MaroBinaryCombine()

            # Specified output file for result.
            combine.open("/path/to/output.bin")

            # Specified 2 binary files to combine.
            combine.combine("/path/to/binary.bin", "/path/to/another.bin")

            # Close the outut file to finish combining.
            combine.close()
    """
    cdef:
        BinaryCombine _combine