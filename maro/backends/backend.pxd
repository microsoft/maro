# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

from cpython cimport bool

from libc.stdint cimport int32_t, int64_t, int16_t, int8_t

# common types

ctypedef unsigned int UINT
ctypedef unsigned int64_t ULONG

# ID of node and attribute
ctypedef unsigned int64_t IDENTIFIER
# Index type of node
ctypedef unsigned int64_t NODE_INDEX
# Index type of slot
ctypedef unsigned int64_t SLOT_INDEX


# Base of all snapshot accessing implementation
cdef class SnapshotListAbc:
    # Query states from snapshot list
    cdef object[object, ndim=1] query(self, str node_name, list ticks, list node_index_list, list attr_list)

    # Record current backend state into snapshot list
    cdef void take_snapshot(self, UINT frame_index) except *

    # List of available frame index in snapshot list
    cdef list get_frame_index_list(self)

    # Enable history, history will dump backend into files each time take_snapshot called
    cdef void enable_history(self, str history_folder) except *

    # Reset internal states
    cdef void reset(self) except *


# Base of all backend implementation
cdef class BackendAbc:
    cdef:
        public SnapshotListAbc snapshots

    # Add a new node to current backend, with specified number (>=0)
    # Returns an ID of this new node in current backend
    cdef IDENTIFIER add_node(self, str name, UINT number) except +

    # Add a new attribute to specified node (id)
    # Returns an ID of this new attribute for current node (id)
    cdef IDENTIFIER add_attr(self, IDENTIFIER node_id, str attr_name, str dtype, UINT slot_num) except +

    # Set value of specified attribute slot.
    # NOTE: since we already know which node current attribute belongs to, so we just need to specify attribute id
    cdef void set_attr_value(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index, object value)  except *
    
    # Get value of specified attribute slot
    cdef object get_attr_value(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index)

    # Set values of specified slots
    cdef void set_attr_values(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX[:] slot_index, list value)  except *

    # Get values of specified slots
    cdef object[object, ndim=1] get_attr_values(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX[:] slot_indices)

    # Get node definition of backend
    cdef dict get_node_info(self)

    # Setup backend with options
    cdef void setup(self, bool enable_snapshot, UINT total_snapshot, dict options) except *

    # Reset internal states
    cdef void reset(self) except *

    