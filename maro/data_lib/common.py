# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple
from struct import Struct

# binary output and reader version
VERSION = 100

SINGLE_BIN_FILE_TYPE = 1
COMBINE_BIN_FILE_TYPE = 2

# used to share header info between writer and reader, with 31 bytes padding for later using
header_struct = Struct("<4s b I Q I QQ QQ qq")

# binary file header
FileHeader = namedtuple(
    "FileHeader",
    [
        "name", "file_type", "version", "item_count", "item_size", "meta_offset", "meta_size",
        "data_offset", "data_size", "starttime", "endtime"
    ]
)

meta_item_format = "20s2s"


# mapping from meta info pack format string
dtype_pack_map = {
    "i": "i",
    "i4": "i",
    "i2": "h",
    "i8": "q",
    "f": "f",
    "d": "d"
}

dtype_convert_map = {
    "i": int,
    'i2': int,
    'i4': int,
    'i8': int,
    'f': float,
    'd': float
}


# merged file part
# row meta: item_count, key value
merged_row_meta_struct = Struct("<H Q")

merged_row_item_count_struct = Struct("< H")
