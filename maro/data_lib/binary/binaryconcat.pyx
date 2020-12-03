# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++



cdef class MaroBinaryConcat:
    def open(self, out_file: str):
        """Open output file for result.

        Args:
            out_file (str): Path to output file.
        """

        self._concat.open(out_file.encode())

    def add(self, bin_file: str):
        """Add a binary file to concat.
        
        Args:
            bin_file (str): Path to binary file to concat.
        """

        self._concat.add(bin_file.encode())


    def close(self):
        """Close output file."""

        self._concat.close()