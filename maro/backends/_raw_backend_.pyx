# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++
#distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
cimport cython

from cython cimport view
from cython.operator cimport dereference as deref

from cpython cimport bool
from libcpp cimport bool as cppbool
from libcpp.map cimport map

from maro.backends.backend cimport (BackendAbc, SnapshotListAbc, AttributeType,
    INT, UINT, ULONG, NODE_TYPE, ATTR_TYPE, NODE_INDEX, SLOT_INDEX,
    ATTR_CHAR, ATTR_UCHAR, ATTR_SHORT, ATTR_USHORT, ATTR_INT, ATTR_UINT,
    ATTR_LONG, ATTR_ULONG, ATTR_FLOAT, ATTR_DOUBLE)


# Ensure numpy will not crash, as we use numpy as query result
np.import_array()

cdef dict attribute_accessors = {
    AttributeType.Byte: AttributeCharAccessor,
    AttributeType.UByte: AttributeUCharAccessor,
    AttributeType.Short: AttributeShortAccessor,
    AttributeType.UShort: AttributeUShortAccessor,
    AttributeType.Int: AttributeIntAccessor,
    AttributeType.UInt: AttributeUIntAccessor,
    AttributeType.Long: AttributeLongAccessor,
    AttributeType.ULong: AttributeULongAccessor,
    AttributeType.Float: AttributeFloatAccessor,
    AttributeType.Double: AttributeDoubleAccessor,
}

cdef map[string, AttrDataType] attr_type_mapping

attr_type_mapping[AttributeType.Byte] = ACHAR
attr_type_mapping[AttributeType.UByte] = AUCHAR
attr_type_mapping[AttributeType.Short] = ASHORT
attr_type_mapping[AttributeType.UShort] = AUSHORT
attr_type_mapping[AttributeType.Int] = AINT
attr_type_mapping[AttributeType.UInt] = AUINT
attr_type_mapping[AttributeType.Long] = ALONG
attr_type_mapping[AttributeType.ULong] = AULONG
attr_type_mapping[AttributeType.Float] = AFLOAT
attr_type_mapping[AttributeType.Double] = ADOUBLE


# Helpers used to access attribute with different data type to avoid to much if-else.
cdef class AttributeAccessor:
    cdef:
        ATTR_TYPE _attr_type
        RawBackend _backend

    cdef void setup(self, RawBackend backend, ATTR_TYPE attr_type):
        self._backend = backend
        self._attr_type = attr_type

    cdef void set_value(self, NODE_INDEX node_index, SLOT_INDEX slot_index, object value) except +:
        pass

    cdef object get_value(self, NODE_INDEX node_index, SLOT_INDEX slot_index) except +:
        pass

    cdef void append_value(self, NODE_INDEX node_index, object value) except +:
        pass

    cdef void insert_value(self, NODE_INDEX node_index, SLOT_INDEX slot_index, object value) except +:
        pass

    def __dealloc__(self):
        self._backend = None

cdef class RawBackend(BackendAbc):
    def __cinit__(self):
        self._node_info = {}
        self._attr_type_dict = {}

    cdef bool is_support_dynamic_features(self):
        return True

    cdef NODE_TYPE add_node(self, str name, NODE_INDEX number) except +:
        cdef NODE_TYPE type = self._frame.add_node(name.encode(), number)

        self._node_info[type] = {"number": number, "name": name, "attrs":{}}

        return type

    cdef ATTR_TYPE add_attr(self, NODE_TYPE node_type, str attr_name, bytes dtype, SLOT_INDEX slot_num, bool is_const, bool is_list) except +:
        cdef AttrDataType dt = AINT

        cdef map[string, AttrDataType].iterator attr_pair = attr_type_mapping.find(dtype)

        if attr_pair != attr_type_mapping.end():
            dt = deref(attr_pair).second;

        # Add attribute to frame.
        cdef ATTR_TYPE attr_type = self._frame.add_attr(node_type, attr_name.encode(), dt, slot_num, is_const, is_list)

        # Initial an access wrapper to this attribute.
        cdef AttributeAccessor acc = attribute_accessors[dtype]()

        acc.setup(self, attr_type)

        self._attr_type_dict[attr_type] = acc

        # Record the information for output.
        self._node_info[node_type]["attrs"][attr_type] = {"type": dtype.decode(), "slots": slot_num, "name": attr_name}

        return attr_type

    cdef void set_attr_value(self, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, object value) except +:
        cdef AttributeAccessor acc = self._attr_type_dict[attr_type]

        acc.set_value(node_index, slot_index, value)

    cdef object get_attr_value(self, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index) except +:
        cdef AttributeAccessor acc = self._attr_type_dict[attr_type]

        return acc.get_value(node_index, slot_index)

    cdef void set_attr_values(self, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX[:] slot_index, list value) except +:
        cdef SLOT_INDEX slot
        cdef int index

        for index, slot in enumerate(slot_index):
            self.set_attr_value(node_index, attr_type, slot, value[index])

    cdef list get_attr_values(self, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX[:] slot_indices) except +:
        cdef AttributeAccessor acc = self._attr_type_dict[attr_type]

        cdef SLOT_INDEX slot

        cdef list result = []

        for slot in slot_indices:
            result.append(acc.get_value(node_index, slot))

        return result

    cdef void append_node(self, NODE_TYPE node_type, NODE_INDEX number) except +:
        self._frame.append_node(node_type, number)

    cdef void delete_node(self, NODE_TYPE node_type, NODE_INDEX node_index) except +:
        self._frame.remove_node(node_type, node_index)

    cdef void resume_node(self, NODE_TYPE node_type, NODE_INDEX node_index) except +:
        self._frame.resume_node(node_type, node_index)

    cdef void append_to_list(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        cdef AttributeAccessor acc = self._attr_type_dict[attr_type]

        acc.append_value(index, value)

    cdef void resize_list(self, NODE_INDEX index, ATTR_TYPE attr_type, SLOT_INDEX new_size) except +:
        self._frame.resize_list(index, attr_type, new_size)

    cdef void clear_list(self, NODE_INDEX index, ATTR_TYPE attr_type) except +:
        self._frame.clear_list(index, attr_type)

    cdef void remove_from_list(self, NODE_INDEX index, ATTR_TYPE attr_type, SLOT_INDEX slot_index) except +:
        self._frame.remove_from_list(index, attr_type, slot_index)

    cdef void insert_to_list(self, NODE_INDEX index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, object value) except +:
        cdef AttributeAccessor acc = self._attr_type_dict[attr_type]

        acc.insert_value(index, slot_index, value)

    cdef void reset(self) except +:
        self._frame.reset()

    cdef void setup(self, bool enable_snapshot, USHORT total_snapshot, dict options) except +:
        self._frame.setup()

        if enable_snapshot:
            self.snapshots = RawSnapshotList(self, total_snapshot)

    cdef dict get_node_info(self) except +:
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

    cdef void dump(self, str folder) except +:
        self._frame.dump(folder.encode())


cdef class RawSnapshotList(SnapshotListAbc):
    def __cinit__(self, RawBackend backend, USHORT total_snapshots):
        self._snapshots.setup(&backend._frame)
        self._snapshots.set_max_size(total_snapshots)

    # Query states from snapshot list
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef query(self, NODE_TYPE node_type, list ticks, list node_index_list, list attr_list) except +:
        cdef int index
        cdef ATTR_TYPE attr_type

        # NOTE: format must be changed if NODE_INDEX type changed
        # Node indices parameters passed to raw backend
        cdef NODE_INDEX[:] node_indices = None
        # Tick parameter passed to raw backend
        cdef INT[:] tick_list = None
        # Attribute list cannot be empty, so we just use it to construct parameter
        cdef ATTR_TYPE[:] attr_type_list = view.array(shape=(len(attr_list),), itemsize=sizeof(ATTR_TYPE), format="I")

        # Check and construct node indices list
        if node_index_list is not None and len(node_index_list) > 0:
            node_indices = view.array(shape=(len(node_index_list),), itemsize=sizeof(NODE_INDEX), format="I")

        cdef USHORT ticks_length = len(ticks)

        # Check ticks, and construct if has value
        if ticks is not None and ticks_length > 0:
            tick_list = view.array(shape=(ticks_length,), itemsize=sizeof(INT), format="i")

            for index in range(ticks_length):
                tick_list[index] = ticks[index]
        else:
            ticks_length = self._snapshots.size()

        for index in range(len(node_index_list)):
            node_indices[index] = node_index_list[index]

        for index in range(len(attr_list)):
            attr_type_list[index] = attr_list[index]

        # Calc 1 frame length
        cdef SnapshotQueryResultShape shape = self._snapshots.prepare(node_type, &tick_list[0], ticks_length, &node_indices[0], len(node_indices), &attr_type_list[0], len(attr_type_list))

        cdef size_t result_size = shape.tick_number * shape.max_node_number * shape.attr_number * shape.max_slot_number

        if result_size <= 0:
            self._snapshots.cancel_query()

            return None

        # Result holder
        cdef QUERY_FLOAT[:, :, :, :] result = view.array(shape=(shape.tick_number, shape.max_node_number, shape.attr_number, shape.max_slot_number), itemsize=sizeof(QUERY_FLOAT), format="f")

        # Default result value
        result[:, :, :, :] = np.nan

        # Do query
        self._snapshots.query(&result[0][0][0][0])

        return np.array(result)

    # Record current backend state into snapshot list
    cdef void take_snapshot(self, INT tick) except +:
        self._snapshots.take_snapshot(tick)

    cdef NODE_INDEX get_node_number(self, NODE_TYPE node_type) except +:
        return self._snapshots.get_max_node_number(node_type)

    # List of available frame index in snapshot list
    cdef list get_frame_index_list(self) except +:
        cdef USHORT number = self._snapshots.size()
        cdef INT[:] result = view.array(shape=(number,), itemsize=sizeof(INT), format="i")

        self._snapshots.get_ticks(&result[0])

        return list(result)

    # Enable history, history will dump backend into files each time take_snapshot called
    cdef void enable_history(self, str history_folder) except +:
        pass

    # Reset internal states
    cdef void reset(self) except +:
        self._snapshots.reset()

    cdef void dump(self, str folder) except +:
        self._snapshots.dump(folder.encode())

    def __len__(self):
        return self._snapshots.size()
