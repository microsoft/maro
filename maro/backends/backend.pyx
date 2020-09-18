# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3

from cpython cimport bool

cdef class SnapshotListAbc:
    cdef query(self, str node_name, list ticks, list node_index_list, list attr_name_list):
        pass

    cdef void take_snapshot(self, int tick) except *:
        pass

    cdef void enable_history(self, str history_folder) except *:
        pass

    cdef void reset(self) except *:
        pass

    cdef list get_frame_index_list(self):
        return []

cdef class BackendAbc:

    cdef void add_node(self, str name, int number) except *:
        pass

    cdef void add_attr(self, str node_name, str attr_name, str dtype, int slot_num) except *:
        pass

    cdef void set_attr_value(self, str node_name, int node_index, str attr_name, int slot_index, value)  except *:
        pass

    cdef object get_attr_value(self, str node_name, int node_index, str attr_name, int slot_index):
        pass
        
    cdef void set_attr_values(self, str node_name, int node_index, str attr_name, int[:] slot_index, list value)  except *:
        pass

    cdef object[object, ndim=1] get_attr_values(self, str node_name, int node_index, str attr_name, int[:] slot_indices):
        pass

    cdef void reset(self) except *:
        pass

    cdef void setup(self, bool enable_snapshot, int total_snapshot, dict options) except *:
        pass

    cdef dict get_node_info(self):
        return {}