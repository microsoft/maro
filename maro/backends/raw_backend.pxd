# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

cimport cython

from cpython cimport bool
from libcpp.string cimport string

from maro.backends.backend cimport (BackendAbc, SnapshotListAbc, INT, UINT, ULONG, USHORT, IDENTIFIER, NODE_INDEX, SLOT_INDEX,
    ATTR_BYTE, ATTR_SHORT, ATTR_INT, ATTR_LONG, ATTR_FLOAT, ATTR_DOUBLE)



cdef extern from "raw/common.h" namespace "maro::backends::raw":
    cdef cppclass AttrDataType:
        pass


cdef extern from "raw/common.h" namespace "maro::backends::raw::AttrDataType":
    cdef AttrDataType ABYTE
    cdef AttrDataType ASHORT
    cdef AttrDataType AINT
    cdef AttrDataType ALONG
    cdef AttrDataType AFLOAT
    cdef AttrDataType ADOUBLE


cdef extern from "raw/attribute.cpp":
    pass


cdef extern from "raw/attribute.h" namespace "maro::backends::raw":
    cdef cppclass Attribute:
        pass


cdef extern from "raw/backend.cpp" namespace "maro::backends::raw":
    pass


cdef extern from "raw/backend.h" namespace "maro::backends::raw":
    cdef cppclass Backend:
        IDENTIFIER add_node(string node_name)
        IDENTIFIER add_attr(IDENTIFIER node_id, string attr_name, AttrDataType attr_type, SLOT_INDEX slot_number)
        ATTR_BYTE get_byte(IDENTIFIER att_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
        ATTR_SHORT get_short(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
        ATTR_INT get_int(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
        ATTR_LONG get_long(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
        ATTR_FLOAT get_float(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
        ATTR_DOUBLE get_double(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
        void set_attr_value[T](IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, T value)

        void set_node_number(IDENTIFIER node_id, NODE_INDEX number)
        void setup(bool enable_snapshot, USHORT snapshot_number)
        void reset_frame()
        void reset_snapshots()
        void take_snapshot(INT tick)
        UINT query_one_tick_length(IDENTIFIER node_id, NODE_INDEX node_indices[], UINT node_length, IDENTIFIER attributes[], UINT attr_length)
        void query(ATTR_FLOAT* result, IDENTIFIER node_id, INT ticks[], UINT ticks_length, NODE_INDEX node_indices[], UINT node_length, IDENTIFIER attributes[], UINT attr_length)
        USHORT get_max_snapshot_number()
        USHORT get_valid_tick_number()

cdef class RawBackend(BackendAbc):
    cdef:
        Backend _backend

        # node name -> IDENTFIER
        dict _node2id_dict

        # attr_id -> dtype
        dict _attr_type_dict

        dict _node_info

cdef class RawSnapshotList(SnapshotListAbc):
    cdef:
        RawBackend _backend
