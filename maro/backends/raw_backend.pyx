# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

cimport cython

from cython cimport view
from cpython cimport bool

from maro.backends.backend cimport (BackendAbc, SnapshotListAbc, INT, UINT, ULONG, IDENTIFIER, NODE_INDEX, SLOT_INDEX,
    ATTR_BYTE, ATTR_SHORT, ATTR_INT, ATTR_LONG, ATTR_FLOAT, ATTR_DOUBLE)

ctypedef fused maro_attribute_types:
    ATTR_BYTE
    ATTR_SHORT
    ATTR_INT
    ATTR_LONG
    ATTR_FLOAT
    ATTR_DOUBLE

cdef class RawBackend(BackendAbc):
    def __cinit__(self):
        self._attr_type_dict = {}
        self.snapshots = RawSnapshotList(self)

    cdef IDENTIFIER add_node(self, str name, NODE_INDEX number) except +:
        cdef IDENTIFIER id = self._backend.add_node(name.encode())

        self._backend.set_node_number(id, number)

        return id

    cdef IDENTIFIER add_attr(self, IDENTIFIER node_id, str attr_name, str dtype, SLOT_INDEX slot_num) except +:
        cdef AttrDataType dt = AttrDataType_INT

        # TODO: refactor later
        if dtype == "i" or dtype == "i4":
            dt = AttrDataType_INT
        elif dtype == "i2":
            dt = AttrDataType_SHORT
        elif dtype == "i8":
            dt = AttrDataType_LONG
        elif dtype == "f":
            dt = AttrDataType_FLOAT
        elif dtype == "d":
            dt = AttrDataType_DOUBLE
        
        cdef IDENTIFIER attr_id = self._backend.add_attr(node_id, attr_name.encode(), dt, slot_num)

        self._attr_type_dict[attr_id] = dtype

        return attr_id

    cdef void set_attr_value(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index, object value)  except *:
        cdef str dt = self._attr_type_dict[attr_id]

        if dt == "i" or dt == "i4":
            self._backend.set_attr_value[ATTR_INT](attr_id, node_index, slot_index, value)
        elif dt == "i2":
            self._backend.set_attr_value[ATTR_BYTE](attr_id, node_index, slot_index, value)
        elif dt == "i8":
            self._backend.set_attr_value[ATTR_LONG](attr_id, node_index, slot_index, value)
        elif dt == "f":
            self._backend.set_attr_value[ATTR_FLOAT](attr_id, node_index, slot_index, value)
        elif dt == "d":
            self._backend.set_attr_value[ATTR_DOUBLE](attr_id, node_index, slot_index, value)

    cdef object get_attr_value(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index):
        cdef str dt = self._attr_type_dict[attr_id]

        if dt == "i" or dt == "i4":
            return self._backend.get_int(attr_id, node_index, slot_index)
        elif dt == "i2":
            return self._backend.get_short(attr_id, node_index, slot_index)
        elif dt == "i8":
            return self._backend.get_long(attr_id, node_index, slot_index)
        elif dt == "f":
            return self._backend.get_float(attr_id, node_index, slot_index)
        elif dt == "d":
            return self._backend.get_double(attr_id, node_index, slot_index)
        
    cdef void set_attr_values(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX[:] slot_index, list value)  except *:
        cdef SLOT_INDEX slot
        cdef int index

        for index, slot in enumerate(slot_index):
            self.set_attr_value(node_index, attr_id, slot, value[index])

    cdef list get_attr_values(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX[:] slot_indices):
        cdef str dt = self._attr_type_dict[attr_id]

        cdef SLOT_INDEX slot

        cdef list result = []

        for slot in slot_indices:
            result.append(self.get_attr_value(node_index, attr_id, slot))

    cdef void reset(self) except *:
        self._backend.reset_frame()

    cdef void setup(self, bool enable_snapshot, UINT total_snapshot, dict options) except *:
        self._backend.setup(enable_snapshot, total_snapshot)

    cdef dict get_node_info(self):
        return {}


cdef class RawSnapshotList(SnapshotListAbc):
    def __cinit__(self, RawBackend backend):
        self._backend = backend;

    # Query states from snapshot list
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef query(self, IDENTIFIER node_id, list ticks, list node_index_list, list attr_list):
        cdef int index
        cdef IDENTIFIER attr_id

        # NOTE: format must be changed if NODE_INDEX type changed
        cdef NODE_INDEX[:] node_indices = view.array(shape=(len(node_index_list),), itemsize=sizeof(NODE_INDEX), format="I")
        cdef IDENTIFIER[:] attr_id_list = view.array(shape=(len(attr_list),), itemsize=sizeof(IDENTIFIER), format="I")
        cdef INT[:] tick_list = view.array(shape=(len(ticks),), itemsize=sizeof(UINT), format="I")

        for index in range(len(node_index_list)):
            node_indices[index] = node_index_list[index]

        for index in range(len(attr_list)):
            attr_id_list[index] = attr_list[index]

        for index in range(len(ticks)):
            tick_list[index] = ticks[index]

        cdef UINT per_frame_length = self._backend._backend.query_one_tick_length(node_id, &node_indices[0], len(node_indices), &attr_id_list[0], len(attr_id_list))

        cdef ATTR_FLOAT[:] result = view.array(shape=(per_frame_length * len(ticks), ), itemsize=sizeof(ATTR_FLOAT), format="f")

        self._backend._backend.query(&result[0], node_id, &tick_list[0], len(tick_list), &node_indices[0], len(node_indices), &attr_id_list[0], len(attr_id_list))

        return result


    # Record current backend state into snapshot list
    cdef void take_snapshot(self, UINT tick) except *:
        self._backend._backend.take_snapshot(tick)

    # List of available frame index in snapshot list
    cdef list get_frame_index_list(self):
        return []

    # Enable history, history will dump backend into files each time take_snapshot called
    cdef void enable_history(self, str history_folder) except *:
        pass

    # Reset internal states
    cdef void reset(self) except *:
        self._backend._backend.reset_snapshots()