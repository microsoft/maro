# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++
#distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import os

import numpy as np

cimport cython
cimport numpy as np
from cpython cimport bool

from maro.backends.backend cimport (
    ATTR_TYPE,
    INT,
    NODE_INDEX,
    NODE_TYPE,
    SLOT_INDEX,
    UINT,
    ULONG,
    USHORT,
    AttributeType,
    BackendAbc,
    SnapshotListAbc,
)

# Attribute data type mapping.
attribute_type_mapping = {
    AttributeType.Byte: "b",
    AttributeType.UByte: "B",
    AttributeType.Short: "h",
    AttributeType.UShort: "H",
    AttributeType.Int: "i",
    AttributeType.UInt: "I",
    AttributeType.Long: "q",
    AttributeType.ULong: "Q",
    AttributeType.Float: "f",
    AttributeType.Double: "d"
}

attribute_type_range = {
    "b": ("AttributeType.Byte", -128, 127),
    "B": ("AttributeType.UByte", 0, 255),
    "h": ("AttributeType.Short", -32768, 32767),
    "H": ("AttributeType.UShort", 0, 65535),
    "i": ("AttributeType.Int", -2147483648, 2147483647),
    "I": ("AttributeType.UInt", 0, 4294967295),
    "q": ("AttributeType.Long", -9223372036854775808, 9223372036854775807),
    "Q": ("AttributeType.ULong", 0, 18446744073709551615),
}


IF NODES_MEMORY_LAYOUT == "ONE_BLOCK":
    # with this flag, we will allocate a big enough memory for all node types, then use this block construct numpy array
    from cpython cimport Py_INCREF, PyObject, PyTypeObject
    from cpython.mem cimport PyMem_Free, PyMem_Malloc
    from libc.string cimport memset

    # we need this to avoid seg fault
    np.import_array()

    # declaration of numpy functions
    cdef extern from "numpy/arrayobject.h":
        PyTypeObject PyArray_Type

        np.ndarray PyArray_SimpleNewFromData(int nd, np.npy_intp * dims, int typenum, void* data)

        np.ndarray PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr, int nd, np.npy_intp* dims, np.npy_intp* strides, void* data, int flags, object obj)



cdef int MMAP_BUFFER_SIZE = 100


# TODO:
# 1. add dtype in header, to make it easy to read
# 2. add tick for each snapshot
cdef class NPBufferedMmap:
    """Buffered memmap of numpy array, as it have size limitation around 2G"""
    def __cinit__(self, str path, np.dtype dtype, int node_number):
        self._node_number = node_number
        self._offset = 0
        self._current_record_number = 0
        self._path = path
        self._dtype = dtype
        self._buffer_size = MMAP_BUFFER_SIZE

        # first time, we will write from start
        self.reload()

    def record(self, np.ndarray arr):
        """Record specified array into file"""
        self._data_arr[self._current_record_number] = arr

        self._current_record_number += 1

        # reload the file with offset if reach the max size
        if self._current_record_number >= self._buffer_size:
            self.reload()

    cdef void reload(self) except +:
        """Reload the file with offset to avoid memmap size limitation"""
        self._data_arr = np.memmap(self._path, self._dtype, "w+", offset=self._offset, shape=(self._buffer_size, self._node_number))

        self._offset += self._dtype.itemsize * self._buffer_size * self._node_number


cdef class NodeInfo:
    """Internal structure to hold node info."""
    cdef:
        public NODE_TYPE type
        public str name
        public NODE_INDEX number

    def __cinit__(self, str name, NODE_TYPE type, NODE_INDEX number):
        self.name = name
        self.type = type
        self.number = number

    def __repr__(self):
        return f"<NodeInfo name: {self.name}, type: {self.type}, number: {self.number}>"


cdef class AttrInfo:
    """Internal structure to hold attribute info"""
    cdef:
        public str name
        public str dtype
        public ATTR_TYPE type
        public NODE_TYPE node_type
        public SLOT_INDEX slot_number

    def __cinit__(self, str name, ATTR_TYPE type, NODE_TYPE node_type, str dtype, SLOT_INDEX slot_number):
        self.name = name
        self.dtype = dtype
        self.type = type
        self.node_type = node_type
        self.slot_number = slot_number

    def gen_numpy_dtype(self):
        """Generate numpy data type (structured)"""
        if self.slot_number == 1:
            return (self.name, self.dtype)
        else:
            return (self.name, self.dtype, self.slot_number)

    def __repr__(self):
        return f"<AttrInfo name: {self.name}, type: {self.type}, node_type: {self.node_type}, slot_number: {self.slot_number}>"

cdef class NumpyBackend(BackendAbc):
    def __cinit__(self):
        self._nodes_list = []
        self._attrs_list = []
        self._node_attr_dict = {}
        self._node_data_dict = {}

    cdef NODE_TYPE add_node(self, str name, NODE_INDEX number) except +:
        """Add a new node type with name and number in backend"""
        cdef NodeInfo new_node = NodeInfo(name, len(self._nodes_list), number)

        self._nodes_list.append(new_node)
        self._node_attr_dict[new_node.type] = []

        return new_node.type

    cdef ATTR_TYPE add_attr(self, NODE_TYPE node_type, str attr_name, bytes dtype, SLOT_INDEX slot_num, bool is_const, bool is_list) except +:
        """Add a new attribute for specified node with data type and slot number"""
        if node_type >= len(self._nodes_list):
            raise Exception("Invalid node type.")

        cdef str _dtype = attribute_type_mapping[dtype]

        cdef NodeInfo node = self._nodes_list[node_type]
        cdef AttrInfo new_attr = AttrInfo(attr_name, len(self._attrs_list), node.type, dtype.decode(), slot_num)

        self._attrs_list.append(new_attr)
        self._node_attr_dict[node_type].append(new_attr)

        return new_attr.type

    cdef void set_attr_value(self, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, object value) except +:
        """Set specified attribute value"""
        if attr_type >= len(self._attrs_list):
            raise Exception("Invalid attribute type.")

        cdef AttrInfo attr = self._attrs_list[attr_type]

        cdef bytes dtype = attr.dtype.encode()
        if dtype in attribute_type_range:
            assert value >= attribute_type_range[dtype][1] and value <= attribute_type_range[dtype][2], (
                f"Value {value} out of range ({attribute_type_range[dtype][0]}: "
                f"[{attribute_type_range[dtype][1]}, {attribute_type_range[dtype][2]}])"
            )

        if attr.node_type >= len(self._nodes_list):
            raise Exception("Invalid node type.")

        cdef NodeInfo node = self._nodes_list[attr.node_type]

        if node_index >= node.number:
            raise Exception("Invalid node index.")

        cdef np.ndarray attr_array = self._node_data_dict[attr.node_type][attr.name]

        if attr.slot_number > 1:
            attr_array[0][node_index, slot_index] = value
        else:
            attr_array[0][node_index] = value

    cdef object get_attr_value(self, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index) except +:
        """Get specified attribute value"""
        if attr_type >= len(self._attrs_list):
            raise Exception("Invalid attribute type.")

        cdef AttrInfo attr = self._attrs_list[attr_type]

        if attr.node_type >= len(self._nodes_list):
            raise Exception("Invalid node type.")

        cdef NodeInfo node = self._nodes_list[attr.node_type]

        if node_index >= node.number:
            raise Exception("Invalid node index.")

        cdef np.ndarray attr_array = self._node_data_dict[attr.node_type][attr.name]

        if attr.slot_number > 1:
            return attr_array[0][node_index, slot_index]
        else:
            return attr_array[0][node_index]

    cdef void set_attr_values(self, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX[:] slot_index, list value) except +:
        cdef AttrInfo attr = self._attrs_list[attr_type]
        cdef np.ndarray attr_array = self._node_data_dict[attr.node_type][attr.name]

        cdef bytes dtype = attr.dtype.encode()

        if attr.slot_number == 1:
            if dtype in attribute_type_range:
                assert value[0] >= attribute_type_range[dtype][1] and value[0] <= attribute_type_range[dtype][2], (
                    f"Value {value[0]} out of range ({attribute_type_range[dtype][0]}: "
                    f"[{attribute_type_range[dtype][1]}, {attribute_type_range[dtype][2]}])"
                )
            attr_array[0][node_index, slot_index[0]] = value[0]
        else:
            if dtype in attribute_type_range:
                for val in value:
                    assert val >= attribute_type_range[dtype][1] and val <= attribute_type_range[dtype][2], (
                        f"Value {val} out of range ({attribute_type_range[dtype][0]}: "
                        f"[{attribute_type_range[dtype][1]}, {attribute_type_range[dtype][2]}])"
                    )
            attr_array[0][node_index, slot_index] = value

    cdef list get_attr_values(self, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX[:] slot_indices) except +:
        cdef AttrInfo attr = self._attrs_list[attr_type]
        cdef np.ndarray attr_array = self._node_data_dict[attr.node_type][attr.name]

        if attr.slot_number == 1:
            return attr_array[0][node_index, slot_indices[0]].tolist()
        else:
            return attr_array[0][node_index, slot_indices].tolist()

    cdef void setup(self, bool enable_snapshot, USHORT total_snapshot, dict options) except +:
        """Set up the numpy backend"""
        self._is_snapshot_enabled = enable_snapshot

        cdef UINT snapshot_number = 0
        cdef str node_name
        cdef NODE_TYPE node_type
        cdef ATTR_TYPE attr_type
        cdef list node_attrs
        cdef np.dtype data_type
        cdef UINT node_number
        cdef AttrInfo ai
        cdef NodeInfo ni
        cdef tuple shape
        cdef UINT max_tick = 0

        IF NODES_MEMORY_LAYOUT == "ONE_BLOCK":
            # Total memory size we need to hold nodes in both frame and snapshot list
            self._data_size = 0
            # Temp node information, as we need several steps to build backend
            node_info = {}

        for node_type, node_attrs in self._node_attr_dict.items():
            ni = self._nodes_list[node_type]

            node_number = ni.number

            data_type = np.dtype([ai.gen_numpy_dtype() for ai in node_attrs])

            # for each node, we keep frame and snapshot in one big numpy array
            # 1st slot is the node's frame data
            # 1-end: are for snapshot list
            if enable_snapshot:
                snapshot_number = total_snapshot

                # first row will be current frame, 1..-1 will be the snapshots
                shape = (snapshot_number + 1, node_number)
            else:
                shape = (1, node_number)

            IF NODES_MEMORY_LAYOUT == "ONE_BLOCK":
                # for ONE_BLOCK mode, we only calculate total size we need to allocate memory
                # shape, data type, beginning of this node
                # NOTE: we have to keep data type here, or it will be collected by GC at sometime,
                # then will cause numpy array cannot get the reference
                # , we will increase he reference later
                node_info[node_type] = (shape, data_type, self._data_size)

                self._data_size += shape[0] * shape[1] * data_type.itemsize
            ELSE:
                # one memory block for each node
                self._node_data_dict[node_type] = np.zeros(shape, data_type)

        IF NODES_MEMORY_LAYOUT == "ONE_BLOCK":
            # allocate memory, and construct numpy array with numpy c api
            self._data = <char*>PyMem_Malloc(self._data_size)

            # TODO: memory allocation failed checking

            # this is much faster to clear than numpy operations
            memset(self._data, 0, self._data_size)

            cdef int offset
            cdef np.npy_intp np_dims[2]

            for node_type, info in node_info.items():
                shape = info[0]
                data_type = info[1]
                offset = info[2]

                np_dims[0] = shape[0]
                np_dims[1] = shape[1]

                self._node_data_dict[node_type] = PyArray_NewFromDescr(&PyArray_Type, data_type, 2, np_dims, NULL, &self._data[offset], np.NPY_ARRAY_C_CONTIGUOUS | np.NPY_ARRAY_WRITEABLE, None)

                # NOTE: we have to increate the reference count of related dtype,
                # or it will cause seg fault
                Py_INCREF(data_type)

        if enable_snapshot:
            self.snapshots = NPSnapshotList(self, snapshot_number + 1)

    def __dealloc__(self):
        """Clear resources before deleted"""
        IF NODES_MEMORY_LAYOUT == "ONE_BLOCK":
            self._node_data_dict = None

            PyMem_Free(self._data)
        ELSE:
            pass

    cdef dict get_node_info(self) except +:
        cdef dict node_info = {}

        cdef NODE_TYPE node_type
        cdef list node_attrs

        for node_type, node_attrs in self._node_attr_dict.items():
            node = self._nodes_list[node_type]

            node_info[node.name] = {
                "number": node.number,
                "attributes": {
                    attr.name: {
                        "type": attr.dtype,
                        "slots": attr.slot_number
                    } for attr in node_attrs
                }
            }

        return node_info


    cdef void reset(self) except +:
        """Reset all the attributes value"""
        cdef NODE_TYPE node_type
        cdef AttrInfo attr_info
        cdef np.ndarray data_arr

        for node_type, data_arr in self._node_data_dict.items():
            # we have to reset by each attribute
            for attr_info in self._node_attr_dict[node_type]:
                # we only reset frame here, without snapshot list
                data_arr[0][attr_info.name] = 0

    cdef void dump(self, str folder) except +:
        for node_type, data_arr in self._node_data_dict.items():
            node = self._nodes_list[node_type]
            node_name = node.name
            filename = os.path.join(folder, node_name + ".npy")
            descFilename = os.path.join(folder, node_name + ".meta")
            with open(filename, "wb+") as f:
                np.save(f, data_arr)
            with open(descFilename, "wt+") as f:
                f.write(",".join([ai.name for ai in self._node_attr_dict[node_type]]) + "\n")
                f.write(",".join([str(ai.slot_number) for ai in self._node_attr_dict[node_type]]))

    cdef list where(self, NODE_INDEX index, ATTR_TYPE attr_type, filter_func: callable) except +:
        cdef AttrInfo attr = self._attrs_list[attr_type]
        cdef np.ndarray attr_array = self._node_data_dict[attr.node_type][attr.name]

        cdef SLOT_INDEX slot_index = 0
        cdef list result_list = []

        for slot_index in range(attr.slot_number):
            if filter_func(attr_array[0][index, slot_index]):
                result_list.append(slot_index)

        return result_list

    cdef list slots_greater_than(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        cdef AttrInfo attr = self._attrs_list[attr_type]
        cdef np.ndarray attr_array = self._node_data_dict[attr.node_type][attr.name]

        return np.where(attr_array[0][index] > value)[0].tolist()

    cdef list slots_greater_equal(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        cdef AttrInfo attr = self._attrs_list[attr_type]
        cdef np.ndarray attr_array = self._node_data_dict[attr.node_type][attr.name]

        return np.where(attr_array[0][index] >= value)[0].tolist()

    cdef list slots_less_than(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        cdef AttrInfo attr = self._attrs_list[attr_type]
        cdef np.ndarray attr_array = self._node_data_dict[attr.node_type][attr.name]

        return np.where(attr_array[0][index] < value)[0].tolist()

    cdef list slots_less_equal(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        cdef AttrInfo attr = self._attrs_list[attr_type]
        cdef np.ndarray attr_array = self._node_data_dict[attr.node_type][attr.name]

        return np.where(attr_array[0][index] <= value)[0].tolist()

    cdef list slots_equal(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        cdef AttrInfo attr = self._attrs_list[attr_type]
        cdef np.ndarray attr_array = self._node_data_dict[attr.node_type][attr.name]

        return np.where(attr_array[0][index] == value)[0].tolist()

    cdef list slots_not_equal(self, NODE_INDEX index, ATTR_TYPE attr_type, object value) except +:
        cdef AttrInfo attr = self._attrs_list[attr_type]
        cdef np.ndarray attr_array = self._node_data_dict[attr.node_type][attr.name]

        return np.where(attr_array[0][index] != value)[0].tolist()

# TODO:
# 1. dump as csv
# 2. take_snapshot(self, bool overwrite_last)
# with this new interface, snapshot will be took sequentially internally without specified tick
# if enable overwrite flat, then last one will be overwrite with latest states, but internal index not change
cdef class NPSnapshotList(SnapshotListAbc):
    """Snapshot list implemented with numpy"""
    def __cinit__(self, NumpyBackend backend, int max_size):
        self._backend = backend

        self._tick2index_dict = {}
        self._index2tick_dict = {}
        self._node_name2type_dict = {}
        self._cur_index = 0
        self._max_size = max_size
        self._is_history_enabled = False
        self._history_dict = {}

        for node in backend._nodes_list:
            self._node_name2type_dict[node.name] = node.type

    cdef NODE_INDEX get_node_number(self, NODE_TYPE node_type) except +:
        cdef NodeInfo node = self._backend._nodes_list[node_type]

        return node.number

    cdef list get_frame_index_list(self) except +:
        return list(self._tick2index_dict.keys())

    cdef void take_snapshot(self, INT tick) except +:
        """Take snapshot for current backend"""
        cdef NODE_TYPE node_type
        cdef NodeInfo ni
        cdef np.ndarray data_arr
        cdef UINT target_index = 0
        cdef INT old_tick # old tick to be removed

        # check if we are overriding exist snapshot, or not inserted yet
        if tick not in self._tick2index_dict:
            self._cur_index += 1

            if self._cur_index >= self._max_size:
                self._cur_index = 1

            target_index = self._cur_index
        else:
            # over-write old one
            target_index = self._tick2index_dict[tick]

        # remove old mapping to make sure _tick2index_dict always keep correct ticks
        if target_index in self._index2tick_dict:
            old_tick = self._index2tick_dict[target_index]

            if old_tick in self._tick2index_dict:
                del self._tick2index_dict[old_tick]

        # recording will copy data at 1st row into _cur_index row
        for node_type, data_arr in self._backend._node_data_dict.items():
            ni = self._backend._nodes_list[node_type]
            data_arr[target_index] = data_arr[0]

            if self._is_history_enabled:
                self._history_dict[ni.name].record(data_arr[0])

        self._index2tick_dict[target_index] = tick

        self._tick2index_dict[tick] = target_index

    cdef query(self, NODE_TYPE node_type, list ticks, list node_index_list, list attr_list) except +:
        cdef UINT tick
        cdef NODE_INDEX node_index
        cdef ATTR_TYPE attr_type
        cdef AttrInfo attr
        cdef np.ndarray data_arr = self._backend._node_data_dict[node_type]

        # TODO: how about use a pre-allocate np array instead concat?
        cdef list retq = []

        if len(ticks) == 0:
            ticks = [t for t in self._tick2index_dict.keys()][-(self._max_size-1):]

        if len(node_index_list) == 0:
            node_index_list = [i for i in range(self._backend._nodes_list[node_type].number)]

        # querying by tick attribute
        for tick in ticks:
            for node_index in node_index_list:
                for attr_type in attr_list:
                    attr = self._backend._attrs_list[attr_type]

                    # since we have a clear tick to index mapping, do not need additional checking here
                    if tick in self._tick2index_dict:
                        retq.append(data_arr[attr.name][self._tick2index_dict[tick], node_index].astype(np.double).flatten())
                    else:
                        # padding for tick which not exist
                        retq.append(np.zeros(attr.slot_number, dtype=np.double))

        return np.concatenate(retq)

    cdef void enable_history(self, str history_folder) except +:
        """Enable history recording, used to save all the snapshots into file"""
        if self._is_history_enabled:
            return

        self._is_history_enabled = True

        cdef NODE_TYPE node_type
        cdef NodeInfo ni
        cdef str dump_path
        cdef np.ndarray data_arr

        for node_type, data_arr in self._backend._node_data_dict.items():
            ni = self._backend._nodes_list[node_type]
            dump_path = os.path.join(history_folder, f"{ni.name}.bin")

            self._history_dict[ni.name] = NPBufferedMmap(dump_path, data_arr.dtype, ni.number)

    cdef void reset(self) except +:
        """Reset snapshot list"""
        self._cur_index = 0
        self._tick2index_dict.clear()
        self._index2tick_dict.clear()
        self._history_dict.clear()

        cdef NODE_TYPE node_type
        cdef AttrInfo attr_info
        cdef np.ndarray data_arr

        for node_type, data_arr in self._backend._node_data_dict.items():
            # we have to reset by each attribute
            for attr_info in self._backend._node_attr_dict[node_type]:
                # we only reset frame here, without snapshot list
                data_arr[1:][attr_info.name] = 0

        # NOTE: we do not reset the history file here, so the file will keep increasing

    def __len__(self):
        return len(self._index2tick_dict)
