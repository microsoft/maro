# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++
import numpy as np
cimport numpy as np
cimport cython

from cython cimport view
from cpython cimport bool

from maro.backends.backend cimport (BackendAbc, SnapshotListAbc, INT, UINT, ULONG, IDENTIFIER, NODE_INDEX, SLOT_INDEX,
    ATTR_BYTE, ATTR_SHORT, ATTR_INT, ATTR_LONG, ATTR_FLOAT, ATTR_DOUBLE)


np.import_array()

ctypedef fused maro_attribute_types:
    ATTR_BYTE
    ATTR_SHORT
    ATTR_INT
    ATTR_LONG
    ATTR_FLOAT
    ATTR_DOUBLE

cdef class RawBackend(BackendAbc):
    def __cinit__(self):
        self._node_info = {}
        self._attr_type_dict = {}
        self.snapshots = RawSnapshotList(self)

    cdef IDENTIFIER add_node(self, str name, NODE_INDEX number) except +:
        cdef IDENTIFIER id = self._backend.add_node(name.encode())

        self._backend.set_node_number(id, number)

        self._node_info[id] = {"number": number, "name": name, "attrs":{}}

        return id

    cdef IDENTIFIER add_attr(self, IDENTIFIER node_id, str attr_name, str dtype, SLOT_INDEX slot_num) except +:
        cdef AttrDataType dt = AINT

        # TODO: refactor later
        if dtype == "i" or dtype == "i4":
            dt = AINT
        elif dtype == "i2":
            dt = ASHORT
        elif dtype == "i8":
            dt = ALONG
        elif dtype == "f":
            dt = AFLOAT
        elif dtype == "d":
            dt = ADOUBLE

        cdef IDENTIFIER attr_id = self._backend.add_attr(node_id, attr_name.encode(), dt, slot_num)

        self._attr_type_dict[attr_id] = dtype

        self._node_info[node_id]["attrs"][attr_id] = {"type": dtype, "slots": slot_num, "name": attr_name}


        return attr_id

    cdef void set_attr_value(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index, object value)  except *:

    cdef void set_attr_value(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index, object value)  except *:
        cdef str dt = self._attr_type_dict[attr_id]

        a = &self._backend.set_attr_value[ATTR_INT]

        # Any good way to avoid this?
        # TODO: pass PythonObject later
        if dt == "i" or dt == "i4":
            self._backend.set_attr_value[ATTR_INT](attr_id, node_index, slot_index, value)
        elif dt == "i2":
            self._backend.set_attr_value[ATTR_SHORT](attr_id, node_index, slot_index, value)
        elif dt == "i8":
            self._backend.set_attr_value[ATTR_LONG](attr_id, node_index, slot_index, value)
        elif dt == "f":
            self._backend.set_attr_value[ATTR_FLOAT](attr_id, node_index, slot_index, value)
        elif dt == "d":
            self._backend.set_attr_value[ATTR_DOUBLE](attr_id, node_index, slot_index, value)

    cdef object get_attr_value(self, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index):
        cdef str dt = self._attr_type_dict[attr_id]

        # TODO: pass PythonObject later
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

        return result

    cdef void reset(self) except *:
        self._backend.reset_frame()

    cdef void setup(self, bool enable_snapshot, USHORT total_snapshot, dict options) except *:
        self._backend.setup(enable_snapshot, total_snapshot)

    cdef dict get_node_info(self):
        cdef dict node_info = {}

        for node_id, node in self._node_info.items():
            node_info[node["name"]] = {
                "number": node["number"],
                "attributes": {
                    attr["name"]: {
                        "type": attr["type"],
                        "slots": attr["slots"]
                    } for _, attr in node["attrs"].items()
                }
            }

        return node_info


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
        cdef NODE_INDEX[:] node_indices = None

        if node_index_list is not None and len(node_index_list) > 0:
            node_indices = view.array(shape=(len(node_index_list),), itemsize=sizeof(NODE_INDEX), format="H")

        cdef IDENTIFIER[:] attr_id_list = view.array(shape=(len(attr_list),), itemsize=sizeof(IDENTIFIER), format="H")

        cdef INT[:] tick_list = None

        cdef USHORT ticks_length = len(ticks)

        if ticks is not None and ticks_length > 0:
            tick_list = view.array(shape=(ticks_length,), itemsize=sizeof(INT), format="i")

            for index in range(ticks_length):
                tick_list[index] = ticks[index]
        else:
            ticks_length = self._backend._backend.get_valid_tick_number()

        for index in range(len(node_index_list)):
            node_indices[index] = node_index_list[index]

        for index in range(len(attr_list)):
            attr_id_list[index] = attr_list[index]

        # Use 1st node to calc frame length
        cdef UINT per_frame_length = self._backend._backend.query_one_tick_length(node_id, &node_indices[0], len(node_indices), &attr_id_list[0], len(attr_id_list))

        cdef ATTR_FLOAT[:] result = view.array(shape=(per_frame_length * ticks_length, ), itemsize=sizeof(ATTR_FLOAT), format="f")

        result[:] = 0

        self._backend._backend.query(&result[0], node_id, &tick_list[0], ticks_length, &node_indices[0], len(node_indices), &attr_id_list[0], len(attr_id_list))

        return np.array(result)

    # Record current backend state into snapshot list
    cdef void take_snapshot(self, INT tick) except *:
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

    def __len__(self):
        return self._backend._backend.get_max_snapshot_number()
