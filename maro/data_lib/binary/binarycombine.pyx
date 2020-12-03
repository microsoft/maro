# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++


cdef class MaroBinaryCombine:
    def open(self, out_file: str):
        """Open out file for result.

        Args:
            out_file (str): Path to output file.
        """
        self._combine.open(out_file)

    def close(self):
        """Close output file."""
        self._combine.close()

    def combine(self, file1: str, file2: str):
        """Combine input binary files into one by timestamp.

        Args:
            file1 (str): Path to 1st file to combine.
            file2 (str): Path to 2nd file to combine.
        """
        self._combine.combine(file1, file2)