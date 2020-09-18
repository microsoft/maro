# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3

from cpython cimport bool

cdef class SnapshotListAbc:
    # query states from snapshot list
    cdef query(self, str node_name, list ticks, list node_index_list, list attr_name_list)

    # record specified backend state into snapshot list
    cdef void take_snapshot(self, int frame_index) except *

    # list of available frame index in snapshot list
    cdef list get_frame_index_list(self)

    cdef void enable_history(self, str history_folder) except *

    cdef void reset(self) except *

cdef class BackendAbc:
    cdef:
        public SnapshotListAbc snapshots

    cdef void setup(self, bool enable_snapshot, int total_snapshot, dict options) except *

    cdef void reset(self) except *

    cdef void add_node(self, str name, int number) except *

    cdef void add_attr(self, str node_name, str attr_name, str dtype, int slot_num) except *

    cdef void set_attr_value(self, str node_name, int node_index, str attr_name, int slot_index, value)  except *

    cdef void set_attr_values(self, str node_name, int node_index, str attr_name, int[:] slot_index, list value)  except *

    cdef object get_attr_value(self, str node_name, int node_index, str attr_name, int slot_index)

    cdef object[object, ndim=1] get_attr_values(self, str node_name, int node_index, str attr_name, int[:] slot_indices)

    cdef dict get_node_info(self)