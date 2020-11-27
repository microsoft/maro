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
        BinaryWriter() except +

        void open(string output_file, string file_type, int32_t file_version) except +

        void close() except +

        void load_meta(string meta_file) except +

        void add_csv(string csv_file) except +

        void set_start_timestamp(ULONGLONG start_timestamp)

cdef class MaroBinaryConverter:
    """Converter that used to convert CSV files with specified META (in toml format).

    NOTE:
        Converter does not care about the order in CSV files, it just convert row by row, so do make sure the input order is fit your
        requirement.

    About meta.toml:
        Meta used to specified which columns need to extract from csv, then convert it into specified data type with an alias into
        binary file.

        Usually meta contains following items:

        1. utc_offset (optional): Used to specified timezone offset of the datetime in CSV files, converter will try to convert all datetime into
            UTC timestamp, default is 0 (means UTC) if not provided.
        2. format (optional): Used to specified the datetime format in CSV files, converter will use this to parse the datetime columns, default is
            '%Y-%m-%D %H-%M-%S' if not provided.
        3. [[row]] (required): A table that contains several column converting information.
        4. [item].alias: Alias of a column, usually used to convert a column name into valid variable name.
        5. [item].column: Name of a column, used to specified which column to convert.
        6. [item].type: Specified target data type this column will converted to.

    Supported data type:
        1.  b: char
        2.  B: unsigned char
        3.  s: short
        4.  S: unsigned short
        5.  i: int32_t
        6.  I: uint32_t
        7.  l: long long
        8.  L: unsigned long long
        9.  f: float
        10. d: double
        11. t: datetime, this is a special data type, it tell converter that current column contains datetime string, we need to do parsing,
            then convert it into timestamp (unsigned long long).

    When converting csv files, we should open it first with output file name, load target meta file, then add csv files.
    Optionally you can specified start timestamp in binary file, or it will use timestamp of first time as start. Usually business engine use
    start timestamp as tick=0, so this would be useful if you need to align the tick to a spcial time point.

    .. code-block:: python

        converter = MaroBinaryConverter()

        # Our target output file
        converter.open("path/to/output/file.bin")

        # Load meta
        converter.load_meta("path/to/meta.toml")

        # Set start timestamp (UTC in seconds)
        converter.set_start_timestamp(1000000)

        # Then add csv files to convert
        for csv_file in [ ... ]:
            converter.add_csv(csv_file)

        # Close the file to flush buffer
        converter.close()
    """
    cdef:
        BinaryWriter _writer
