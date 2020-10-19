# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

from cpython cimport bool


cdef class SnapshotListAbc:
    cdef object[object, ndim=1] query(self, str node_name, list ticks, list node_index_list, list attr_list):
        pass

    cdef void take_snapshot(self, UINT frame_index) except *:
        pass

    cdef void enable_history(self, str history_folder) except *:
        pass

    cdef void reset(self) except *:
        pass

    cdef list get_frame_index_list(self):
        return []


cdef class BackendAbc:

    cdef IDENTIFIER add_node(self, str name, UINT number) except +:
        pass

    cdef IDENTIFIER add_attr(self, IDENTIFIER node_id, str attr_name, str dtype, UINT slot_num) except +:
        pass

    cdef void set_attr_value(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index, object value)  except *:
        pass

    cdef object get_attr_value(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index):
        pass
        
    cdef void set_attr_values(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX[:] slot_index, list value)  except *:
        pass

    cdef object[object, ndim=1] get_attr_values(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX[:] slot_indices):
        pass

    cdef void reset(self) except *:
        pass

    cdef void setup(self, bool enable_snapshot, int total_snapshot, dict options) except *:
        pass

    cdef dict get_node_info(self):
        return {}