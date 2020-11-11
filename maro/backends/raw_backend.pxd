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


cdef extern from "raw/bitset.h" namespace "maro::backends::raw":
    cdef cppclass Bitset:
        pass


cdef extern from "raw/bitset.cpp" namespace "maro::backends::raw":
    pass


cdef extern from "raw/attributestore.h" namespace "maro::backends::raw":
    cdef cppclass AttributeStore:
        pass


cdef extern from "raw/attributestore.cpp" namespace "maro::backends::raw":
    pass


cdef extern from "raw/snapshotlist.h" namespace "maro::backends::raw":
    cdef cppclass SnapshotList:
        pass

    cdef struct SnapshotResultShape:
        INT tick_number
        NODE_INDEX max_node_number
        USHORT attr_number
        SLOT_INDEX max_slot_number


cdef extern from "raw/snapshotlist.cpp" namespace "maro::backends::raw":
    pass


cdef extern from "raw/frame.h" namespace "maro::backends::raw":
    cdef cppclass Frame:
        pass


cdef extern from "raw/frame.cpp" namespace "maro::backends::raw":
    pass


cdef extern from "raw/backend.cpp" namespace "maro::backends::raw":
    pass


cdef extern from "raw/backend.h" namespace "maro::backends::raw":
    cdef cppclass Backend:
        IDENTIFIER add_node(string node_name, NODE_INDEX node_num)
        IDENTIFIER add_attr(IDENTIFIER node_id, string attr_name, AttrDataType attr_type, SLOT_INDEX slot_number)

        ATTR_BYTE get_byte(IDENTIFIER att_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
        ATTR_SHORT get_short(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
        ATTR_INT get_int(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
        ATTR_LONG get_long(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
        ATTR_FLOAT get_float(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
        ATTR_DOUBLE get_double(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
        void set_attr_value[T](IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, T value)

        void setup(bool enable_snapshot, USHORT snapshot_number)

        void reset_frame()
        void reset_snapshots()

        void take_snapshot(INT tick)

        SnapshotResultShape prepare(IDENTIFIER node_id, INT ticks[], UINT tick_length, NODE_INDEX node_indices[], UINT node_length, IDENTIFIER attributes[], UINT attr_length)
        void query(ATTR_FLOAT* result, SnapshotResultShape shape)

        USHORT get_max_snapshot_number()
        USHORT get_valid_tick_number()

        void get_ticks(INT *result)

        void dump(string path)

        void append_node(IDENTIFIER node_id, NODE_INDEX number)
        void delete_node(IDENTIFIER node_id, NODE_INDEX node_index)
        void resume_node(IDENTIFIER node_id, NODE_INDEX node_index)
        void set_attribute_slot(IDENTIFIER attr_id, SLOT_INDEX slots)
        USHORT get_node_number(IDENTIFIER node_id)
        USHORT get_slots_number(IDENTIFIER attr_id)


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
        RawBackend _raw
