# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

cimport cython
from cpython cimport bool
from libcpp cimport bool as cppbool
from libcpp.string cimport string

from maro.backends.backend cimport (
    ATTR_CHAR,
    ATTR_DOUBLE,
    ATTR_FLOAT,
    ATTR_INT,
    ATTR_LONG,
    ATTR_SHORT,
    ATTR_TYPE,
    INT,
    NODE_INDEX,
    NODE_TYPE,
    QUERY_FLOAT,
    SLOT_INDEX,
    UINT,
    ULONG,
    USHORT,
    BackendAbc,
    SnapshotListAbc,
)


cdef extern from "raw/common.h" namespace "maro::backends::raw":
    cdef cppclass AttrDataType:
        pass


cdef extern from "raw/common.h" namespace "maro::backends::raw::AttrDataType":
    cdef AttrDataType ACHAR
    cdef AttrDataType AUCHAR
    cdef AttrDataType ASHORT
    cdef AttrDataType AUSHORT
    cdef AttrDataType AINT
    cdef AttrDataType AUINT
    cdef AttrDataType ALONG
    cdef AttrDataType AULONG
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


cdef extern from "raw/node.h" namespace "maro::backends::raw":
    cdef cppclass Node:
        pass


cdef extern from "raw/node.cpp" namespace "maro::backends::raw":
    pass


cdef extern from "raw/frame.h" namespace "maro::backends::raw":
    cdef cppclass Frame:
        Frame()
        Frame(const Frame& frame)
        Frame& operator=(const Frame& frame)

        NODE_TYPE add_node(string node_name, NODE_INDEX node_number)
        ATTR_TYPE add_attr(NODE_TYPE node_type, string attr_name, AttrDataType data_type, SLOT_INDEX slot_number, cppbool is_const, cppbool is_list)

        void append_node(NODE_TYPE node_type, NODE_INDEX node_number)
        void resume_node(NODE_TYPE node_type, NODE_INDEX node_number)
        void remove_node(NODE_TYPE node_type, NODE_INDEX node_index)

        T get_value[T](NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index)
        void set_value[T](NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, T value)

        void append_to_list[T](NODE_INDEX node_index, ATTR_TYPE attr_type, T value)
        void clear_list(NODE_INDEX node_index, ATTR_TYPE attr_type)
        void resize_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX new_size)
        void remove_from_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index)
        void insert_to_list[T](NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, T value)
        SLOT_INDEX get_slot_number(NODE_INDEX node_index, ATTR_TYPE attr_type)

        void setup()
        void reset()
        void dump(string path)


cdef extern from "raw/frame.cpp" namespace "maro::backends::raw":
    pass


cdef extern from "raw/snapshotlist.h" namespace "maro::backends::raw":
    cdef cppclass SnapshotList:
        void set_max_size(USHORT max_size)
        void setup(Frame* frame)

        void take_snapshot(int ticks)

        UINT size() const
        UINT max_size() const
        NODE_INDEX get_max_node_number(NODE_TYPE node_type) const

        void reset()

        void dump(string path)

        void get_ticks(int* result) const

        SnapshotQueryResultShape prepare(NODE_TYPE node_type, int ticks[], UINT tick_length, NODE_INDEX node_indices[], UINT node_length, ATTR_TYPE attributes[], UINT attr_length)
        void query(QUERY_FLOAT* result)
        void cancel_query()

    cdef struct SnapshotQueryResultShape:
        USHORT attr_number
        int tick_number
        SLOT_INDEX max_slot_number
        NODE_INDEX max_node_number


cdef extern from "raw/snapshotlist.cpp" namespace "maro::backends::raw":
    pass


cdef class RawBackend(BackendAbc):
    cdef:
        Frame _frame

        # node name -> ATTR_TYPE
        dict _node2type_dict

        # attr_type -> dtype
        dict _attr_type_dict

        dict _node_info


cdef class RawSnapshotList(SnapshotListAbc):
    cdef:
        SnapshotList _snapshots
