# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

from cpython cimport bool

from libc.stdint cimport int32_t, int64_t, int16_t, int8_t, uint32_t, uint64_t

# common types

ctypedef int INT
ctypedef unsigned int UINT
ctypedef unsigned long long ULONG
ctypedef unsigned short USHORT

ctypedef int8_t ATTR_BYTE
ctypedef int16_t ATTR_SHORT
ctypedef int32_t ATTR_INT
ctypedef int64_t ATTR_LONG
ctypedef float ATTR_FLOAT
ctypedef double ATTR_DOUBLE


# ID of node and attribute
ctypedef unsigned short IDENTIFIER
# Index type of node
ctypedef unsigned short NODE_INDEX
# Index type of slot
ctypedef unsigned short SLOT_INDEX


# Base of all snapshot accessing implementation
cdef class SnapshotListAbc:
    # Query states from snapshot list
    cdef query(self, IDENTIFIER node_id, list ticks, list node_index_list, list attr_list) except +

    # Record current backend state into snapshot list
    cdef void take_snapshot(self, INT tick) except +

    # List of available frame index in snapshot list
    cdef list get_frame_index_list(self) except +

    # Get number of specified node
    cdef USHORT get_node_number(self, IDENTIFIER node_id) except +

    # Get number of slots for specified attribute
    cdef USHORT get_slots_number(self, IDENTIFIER attr_id) except +

    # Enable history, history will dump backend into files each time take_snapshot called
    cdef void enable_history(self, str history_folder) except +

    # Reset internal states
    cdef void reset(self) except +


# Base of all backend implementation
cdef class BackendAbc:
    cdef:
        public SnapshotListAbc snapshots

    # Is current backend support dynamic features
    cdef bool is_support_dynamic_features(self)

    # Add a new node to current backend, with specified number (>=0)
    # Returns an ID of this new node in current backend
    cdef IDENTIFIER add_node(self, str name, NODE_INDEX number) except +

    # Add a new attribute to specified node (id)
    # Returns an ID of this new attribute for current node (id)
    cdef IDENTIFIER add_attr(self, IDENTIFIER node_id, str attr_name, str dtype, SLOT_INDEX slot_num) except +

    # Set value of specified attribute slot.
    # NOTE: since we already know which node current attribute belongs to, so we just need to specify attribute id
    cdef void set_attr_value(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index, object value) except +

    # Get value of specified attribute slot
    cdef object get_attr_value(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index) except +

    # Set values of specified slots
    cdef void set_attr_values(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX[:] slot_index, list value) except +

    # Get values of specified slots
    cdef list get_attr_values(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX[:] slot_indices) except +

    # Get node definition of backend
    cdef dict get_node_info(self) except +

    # Setup backend with options
    cdef void setup(self, bool enable_snapshot, USHORT total_snapshot, dict options) except +

    # Reset internal states
    cdef void reset(self) except +

    # Append specified number of nodes
    cdef void append_node(self, IDENTIFIER node_id, NODE_INDEX number) except +

    # Delete a node by index
    cdef void delete_node(self, IDENTIFIER node_id, NODE_INDEX node_index) except +

    # Resume node that been deleted
    cdef void resume_node(self, IDENTIFIER node_id, NODE_INDEX node_index) except +

    # Set slot number of specified attribute
    cdef void set_attribute_slot(self, IDENTIFIER attr_id, SLOT_INDEX slots) except +