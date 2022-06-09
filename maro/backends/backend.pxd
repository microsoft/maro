# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

from cpython cimport bool
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t, uint32_t, uint64_t

# common types

ctypedef int INT
ctypedef unsigned int UINT
ctypedef unsigned long long ULONG
ctypedef unsigned short USHORT

ctypedef char ATTR_CHAR
ctypedef unsigned char ATTR_UCHAR
ctypedef short ATTR_SHORT
ctypedef USHORT ATTR_USHORT
ctypedef int32_t ATTR_INT
ctypedef uint32_t ATTR_UINT
ctypedef int64_t ATTR_LONG
ctypedef uint64_t ATTR_ULONG
ctypedef float ATTR_FLOAT
ctypedef double ATTR_DOUBLE

# Type for snapshot querying.
ctypedef double QUERY_FLOAT

# TYPE of node and attribute
ctypedef unsigned short NODE_TYPE
ctypedef uint32_t ATTR_TYPE

# Index type of node
ctypedef ATTR_TYPE NODE_INDEX

# Index type of slot
ctypedef ATTR_TYPE SLOT_INDEX


cdef class AttributeType:
    pass


# Base of all snapshot accessing implementation
cdef class SnapshotListAbc:
    # Query states from snapshot list
    cdef query(self, NODE_TYPE node_type, list ticks, list node_index_list, list attr_list) except +

    # Record current backend state into snapshot list
    cdef void take_snapshot(self, INT tick) except +

    # List of available frame index in snapshot list
    cdef list get_frame_index_list(self) except +

    # Get number of specified node
    cdef NODE_INDEX get_node_number(self, NODE_TYPE node_type) except +

    # Enable history, history will dump backend into files each time take_snapshot called
    cdef void enable_history(self, str history_folder) except +

    # Reset internal states.
    cdef void reset(self) except +

    # Dump Snapshot into target folder (without filename).
    cdef void dump(self, str folder) except +


# Base of all backend implementation
cdef class BackendAbc:
    cdef:
        public SnapshotListAbc snapshots

    # Is current backend support dynamic features.
    cdef bool is_support_dynamic_features(self)

    # Add a new node to current backend, with specified number (>=0).
    # Returns an ID of this new node in current backend.
    cdef NODE_TYPE add_node(self, str name, NODE_INDEX number) except +

    # Add a new attribute to specified node (id).
    # Returns an ID of this new attribute for current node (id).
    cdef ATTR_TYPE add_attr(self, NODE_TYPE node_type, str attr_name, bytes dtype, SLOT_INDEX slot_num, bool is_const, bool is_list) except +

    # Set value of specified attribute slot.
    # NOTE: since we already know which node current attribute belongs to, so we just need to specify attribute id
    cdef void set_attr_value(self, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, object value) except +

    # Get value of specified attribute slot.
    cdef object get_attr_value(self, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index) except +

    # Set values of specified slots.
    cdef void set_attr_values(self, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX[:] slot_index, list value) except +

    # Get values of specified slots.
    cdef list get_attr_values(self, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX[:] slot_indices) except +

    # Get node definition of backend.
    cdef dict get_node_info(self) except +

    # Setup backend with options.
    cdef void setup(self, bool enable_snapshot, USHORT total_snapshot, dict options) except +

    # Reset internal states.
    cdef void reset(self) except +

    # Append specified number of nodes.
    cdef void append_node(self, NODE_TYPE node_type, NODE_INDEX number) except +

    # Delete a node by index.
    cdef void delete_node(self, NODE_TYPE node_type, NODE_INDEX node_index) except +

    # Resume node that been deleted.
    cdef void resume_node(self, NODE_TYPE node_type, NODE_INDEX node_index) except +

    # Append value to specified list attribute.
    cdef void append_to_list(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +

    # Resize specified list attribute.
    cdef void resize_list(self, NODE_INDEX index, ATTR_TYPE attr_type, SLOT_INDEX new_size) except +

    # Clear specified list attribute.
    cdef void clear_list(self, NODE_INDEX index, ATTR_TYPE attr_type) except +

    # Remove a slot from list attribute.
    cdef void remove_from_list(self, NODE_INDEX index, ATTR_TYPE attr_type, SLOT_INDEX slot_index) except +

    # Insert a slot to list attribute.
    cdef void insert_to_list(self, NODE_INDEX index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, object value) except +

    # Dump Snapshot into target folder (without filename).
    cdef void dump(self, str folder) except +

    # Filter with input function.
    cdef list where(self, NODE_INDEX index, ATTR_TYPE attr_type, filter_func: callable) except +

    # Filter slots that greater than specified value.
    cdef list slots_greater_than(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +

    # Filter slots that greater equal to specified value.
    cdef list slots_greater_equal(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +

    # Filter slots that less than specified value.
    cdef list slots_less_than(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +

    # Filter slots that less equal to specified value.
    cdef list slots_less_equal(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +

    # Filter slots that equal to specified value.
    cdef list slots_equal(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +

    # Filter slots that not equal to specified value.
    cdef list slots_not_equal(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +

    # Get slot number for specified attribute, only support dynamic backend.
    cdef SLOT_INDEX get_slot_number(self, NODE_INDEX index, ATTR_TYPE attr_type) except +
