# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

from enum import Enum

from cpython cimport bool


cdef class AttributeType:
    Byte = b"byte"
    UByte = b"ubyte"
    Short = b"short"
    UShort = b"ushort"
    Int = b"int"
    UInt = b"uint"
    Long = b"long"
    ULong = b"ulong"
    Float = b"float"
    Double = b"double"

cdef int raise_get_attr_error() except +:
    raise Exception("Bad parameters to get attribute value.")


cdef class SnapshotListAbc:
    cdef query(self, NODE_TYPE node_type, list ticks, list node_index_list, list attr_list) except +:
        pass

    cdef void take_snapshot(self, INT tick) except +:
        pass

    cdef NODE_INDEX get_node_number(self, NODE_TYPE node_type) except +:
        return 0

    cdef void enable_history(self, str history_folder) except +:
        pass

    cdef void reset(self) except +:
        pass

    cdef list get_frame_index_list(self) except +:
        return []

    cdef void dump(self, str folder) except +:
        pass

cdef class BackendAbc:

    cdef bool is_support_dynamic_features(self):
        return False

    cdef NODE_TYPE add_node(self, str name, NODE_INDEX number) except +:
        pass

    cdef ATTR_TYPE add_attr(self, NODE_TYPE node_type, str attr_name, bytes dtype, SLOT_INDEX slot_num, bool is_const, bool is_list) except +:
        pass

    cdef void set_attr_value(self, NODE_INDEX node_index, ATTR_TYPE attr_id, SLOT_INDEX slot_index, object value) except +:
        pass

    cdef object get_attr_value(self, NODE_INDEX node_index, ATTR_TYPE attr_id, SLOT_INDEX slot_index) except +:
        pass

    cdef void set_attr_values(self, NODE_INDEX node_index, ATTR_TYPE attr_id, SLOT_INDEX[:] slot_index, list value) except +:
        pass

    cdef list get_attr_values(self, NODE_INDEX node_index, ATTR_TYPE attr_id, SLOT_INDEX[:] slot_indices) except +:
        pass

    cdef void reset(self) except +:
        pass

    cdef void setup(self, bool enable_snapshot, USHORT total_snapshot, dict options) except +:
        pass

    cdef dict get_node_info(self) except +:
        return {}

    cdef void append_node(self, NODE_TYPE node_type, NODE_INDEX number) except +:
        pass

    cdef void delete_node(self, NODE_TYPE node_type, NODE_INDEX node_index) except +:
        pass

    cdef void resume_node(self, NODE_TYPE node_type, NODE_INDEX node_index) except +:
        pass

    cdef void append_to_list(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        pass

    # Resize specified list attribute.
    cdef void resize_list(self, NODE_INDEX index, ATTR_TYPE attr_type, SLOT_INDEX new_size) except +:
        pass

    # Clear specified list attribute.
    cdef void clear_list(self, NODE_INDEX index, ATTR_TYPE attr_type) except +:
        pass

    cdef void remove_from_list(self, NODE_INDEX index, ATTR_TYPE attr_type, SLOT_INDEX slot_index) except +:
        pass

    cdef void insert_to_list(self, NODE_INDEX index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, object value) except +:
        pass

    cdef void dump(self, str folder) except +:
        pass

    cdef list where(self, NODE_INDEX index, ATTR_TYPE attr_type, filter_func: callable) except +:
        pass

    cdef list slots_greater_than(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        pass

    cdef list slots_greater_equal(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        pass

    cdef list slots_less_than(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        pass

    cdef list slots_less_equal(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        pass

    cdef list slots_equal(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        pass

    cdef list slots_not_equal(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        pass

    cdef SLOT_INDEX get_slot_number(self, NODE_INDEX index, ATTR_TYPE attr_type) except +:
        pass
