# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++


cimport cython

from libcpp.string cimport string


cdef extern from "cpp/binaryconcat.cpp":
    pass


cdef extern from "cpp/binaryconcat.h" namespace "maro::datalib":
    cdef cppclass BinaryConcat:
        BinaryConcat()
        void open(string out_file) except +
        void add(string bin_file) except +
        void close() except +


cdef class MaroBinaryConcat:
    """Util used to comcat binary files, append one to the tail of anther one by input order.

        .. code-block:: python

            concat = MaroBinaryConcat()

            # Specified output file for result.
            concat.open("/path/to/output.bin")

            # Add file to concat.
            concat.add("/path/to/binary.bin")

            # Add more files.
            concat.add("/path/to/another.bin")

            # Close the outut file to finish concating.
            concat.close()
    """
    cdef:
        BinaryConcat _concat