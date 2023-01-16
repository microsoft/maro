# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++
#distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import os

cimport cython
cimport numpy as np

import numpy as np

from cpython cimport bool

from typing import Union

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
from maro.backends.np_backend cimport NumpyBackend
from maro.backends.raw_backend cimport RawBackend

from maro.utils.exception.backends_exception import (
    BackendsAccessDeletedNodeException,
    BackendsAppendToNonListAttributeException,
    BackendsArrayAttributeAccessException,
    BackendsClearNonListAttributeException,
    BackendsGetItemInvalidException,
    BackendsInsertNonListAttributeException,
    BackendsInvalidAttributeException,
    BackendsInvalidNodeException,
    BackendsRemoveFromNonListAttributeException,
    BackendsResizeNonListAttributeException,
    BackendsSetItemInvalidException,
)

# Old type definition mapping.
old_data_type_definitions = {
    "i": AttributeType.Int,
    "i4": AttributeType.Int,
    "i2": AttributeType.Short,
    "i8": AttributeType.Long,
    "f": AttributeType.Float,
    "d": AttributeType.Double
}

# Supported backends.
backend_dict = {
    "dynamic" : RawBackend,
    "static" : NumpyBackend
}

# Default backend name.
_default_backend_name = "static"

NP_SLOT_INDEX = np.uint32
NP_NODE_INDEX = np.uint32


def node(name: str):
    """Frame node decorator, used to specified node name in Frame and SnapshotList.

    This node name used for querying from snapshot list, see NodeBase for details.

    Args:
        name(str): Node name in Frame.
    """
    def node_dec(cls):
        cls.__node_name__ = name

        return cls

    return node_dec


def try_get_attribute(target, name, default=None):
    try:
        attr = object.__getattribute__(target, name)

        return attr
    except:
        return default


cdef class NodeAttribute:
    def __cinit__(self, object dtype = None, SLOT_INDEX slot_num = 1, is_const = False, is_list = False):
        # Check the type of dtype, used to compact with old version
        cdef bytes _type = AttributeType.Int

        if dtype is not None:
            dtype_type = type(dtype)

            if dtype_type == str:
                if dtype in old_data_type_definitions:
                    _type = old_data_type_definitions[dtype]
            elif dtype_type == bytes:
                _type = dtype

        self._dtype = _type
        self._slot_number = slot_num
        self._is_const = is_const
        self._is_list = is_list


# Wrapper to provide easy way to access attribute value of specified node
# with this wrapper, user can get/set attribute value more easily.
cdef class _NodeAttributeAccessor:
    cdef:
        # Target node index.
        NODE_INDEX _node_index

        # Target attribute type.
        public ATTR_TYPE _attr_type

        # Slot number of target attribute.
        public SLOT_INDEX _slot_number

        # Is this is a list attribute?
        # True to enable append/remove/insert methods.
        public bool _is_list

        # Target backend.
        BackendAbc _backend

        # Index used to support for-loop
        SLOT_INDEX _cur_iter_slot_index

        # Enable dynamic attributes.
        dict __dict__

    def __cinit__(self, NodeAttribute attr, ATTR_TYPE attr_type, BackendAbc backend, NODE_INDEX node_index):
        self._attr_type = attr_type
        self._node_index = node_index
        self._slot_number = attr._slot_number
        self._is_list = attr._is_list
        self._backend = backend

        # Built-in index too support for-loop
        self._cur_iter_slot_index = 0

        # Special for list attribute, we need to slot number to support __len__
        # We will count the slot number here, though we can get it from function call
        if self._is_list:
            self._slot_number = 0

    def append(self, value):
        """Append a value to current attribute.

        NOTE:
            Current attribute must be a list.

        Args:
            value(object): Value to append, the data type must fit the declared one.
        """
        if not self._is_list:
            raise BackendsAppendToNonListAttributeException()

        self._backend.append_to_list(self._node_index, self._attr_type, value)

        self._slot_number = self._backend.get_slot_number(self._node_index, self._attr_type)

        if "_cb" in self.__dict__:
            self._cb(None)

    def resize(self, new_size: int):
        """Resize current list attribute with specified new size.

        NOTE:
            Current attribute must be a list.

        Args:
            new_size(int): New size to resize, max number is 2^32.
        """
        if not self._is_list:
            raise BackendsResizeNonListAttributeException()

        self._backend.resize_list(self._node_index, self._attr_type, new_size)

        self._slot_number = self._backend.get_slot_number(self._node_index, self._attr_type)

        if "_cb" in self.__dict__:
            self._cb(None)

    def clear(self):
        """Clear all items in current list attribute.

        NOTE:
            Current attribute must be a list.
        """
        if not self._is_list:
            raise BackendsClearNonListAttributeException()

        self._backend.clear_list(self._node_index, self._attr_type)

        self._slot_number = 0

        if "_cb" in self.__dict__:
            self._cb(None)

    def insert(self, slot_index: int, value: object):
        """Insert a value to specified slot.

        Args:
            slot_index(int): Slot index to insert.
            value(object): Value to insert.
        """
        if not self._is_list:
            raise BackendsInsertNonListAttributeException()

        self._backend.insert_to_list(self._node_index, self._attr_type, slot_index, value)

        self._slot_number = self._backend.get_slot_number(self._node_index, self._attr_type)

        if "_cb" in self.__dict__:
            self._cb(None)

    def remove(self, slot_index: int):
        """Remove specified slot.

        Args:
            slot_index(int): Slot index to remove.
        """
        if not self._is_list:
            raise BackendsRemoveFromNonListAttributeException()

        self._backend.remove_from_list(self._node_index, self._attr_type, slot_index)

        self._slot_number = self._backend.get_slot_number(self._node_index, self._attr_type)

        if "_cb" in self.__dict__:
            self._cb(None)

    def where(self, filter_func: callable):
        """Filter current attribute slots with input function.

        Args:
            filter_func (callable): Function to filter slot value.

        Returns:
            List[int]: List of slot index whose value match the filter function.
        """
        return self._backend.where(self._node_index, self._attr_type, filter_func)

    def __lt__(self, other):
        return self._backend.slots_less_than(self._node_index, self._attr_type, other)

    def __le__(self, other):
        return self._backend.slots_less_equal(self._node_index, self._attr_type, other)

    def __gt__(self, other):
        return self._backend.slots_greater_than(self._node_index, self._attr_type, other)

    def __ge__(self, other):
        return self._backend.slots_greater_equal(self._node_index, self._attr_type, other)

    def __eq__(self, other):
        return self._backend.slots_equal(self._node_index, self._attr_type, other)

    def __ne__(self, other):
        return self._backend.slots_not_equal(self._node_index, self._attr_type, other)

    def __iter__(self):
        """Start for-loop."""
        self._cur_iter_slot_index = 0

        return self

    def __next__(self):
        """Get next slot value."""
        if self._cur_iter_slot_index >= self._slot_number:
            raise StopIteration

        value = self._backend.get_attr_value(self._node_index, self._attr_type, self._cur_iter_slot_index)

        self._cur_iter_slot_index += 1

        return value

    def __getitem__(self, slot: Union[int, slice, list, tuple]):
        """We use this function to support slice interface to access attribute, like:

            a = node.attr1[1:]
            b = node.attr1[:]
            c = node.attr1[(1, 2, 3)]
        """
        # NOTE: we do not support negative indexing now

        cdef SLOT_INDEX start
        cdef SLOT_INDEX stop
        cdef type slot_type = type(slot)
        cdef SLOT_INDEX[:] slot_list

        # Get only one slot: node.attribute[0].
        if slot_type == int:
            return self._backend.get_attr_value(self._node_index, self._attr_type, slot)

        # Try to support following:
        # node.attribute[1:3]
        # node.attribute[[1, 2, 3]]
        # node.attribute[(0, 1)]
        cdef tuple slot_key = tuple(slot) if slot_type != slice else (slot.start, slot.stop, slot.step)

        slot_list = None

        # Parse slice parameters: [:].
        if slot_type == slice:
            start = 0 if slot.start is None else slot.start
            stop = self._slot_number if slot.stop is None else slot.stop

            slot_list = np.arange(start, stop, dtype=NP_SLOT_INDEX)
        elif slot_type == list or slot_type == tuple:
            slot_list = np.array(slot, dtype=NP_SLOT_INDEX)
        else:
            raise BackendsGetItemInvalidException()

        return self._backend.get_attr_values(self._node_index, self._attr_type, slot_list)

    def __setitem__(self, slot: Union[int, slice, list, tuple], value: Union[object, list, tuple, np.ndarray]):
        cdef SLOT_INDEX[:] slot_list
        cdef list values

        cdef SLOT_INDEX start
        cdef SLOT_INDEX stop

        cdef type slot_type = type(slot)
        cdef type value_type = type(value)

        cdef SLOT_INDEX values_length
        cdef SLOT_INDEX slot_length
        cdef tuple slot_key

        # Set value for one slot: node.attribute[0] = 1.
        if slot_type == int:
            self._backend.set_attr_value(self._node_index, self._attr_type, slot, value)
        elif slot_type == list or slot_type == tuple or slot_type == slice:
            # Try to support following:
            # node.attribute[0: 2] = 1 / [1,2] / (0, 2, 3)
            slot_key = tuple(slot) if slot_type != slice else (slot.start, slot.stop, slot.step)

            slot_list = None

            # Parse slot indices to set.
            if slot_type == slice:
                start = 0 if slot.start is None else slot.start
                stop = self._slot_number if slot.stop is None else slot.stop

                slot_list = np.arange(start, stop, dtype=NP_SLOT_INDEX)
            elif slot_type == list or slot_type == tuple:
                slot_list = np.array(slot, dtype=NP_SLOT_INDEX)

            slot_length = len(slot_list)

            # Parse value, padding if needed.
            if value_type == list or value_type == tuple or value_type == np.ndarray:
                values = list(value)

                values_length = len(values)

                # Make sure the value size is same as slot size.
                if values_length > slot_length:
                    values = values[0: slot_length]
                elif values_length < slot_length:
                    slot_list = slot_list[0: values_length]
            else:
                values = [value] * slot_length

            self._backend.set_attr_values(self._node_index, self._attr_type, slot_list, values)
        else:
            raise BackendsSetItemInvalidException()

        # Check and invoke value changed callback.
        if "_cb" in self.__dict__:
            self._cb(value)

    def __len__(self):
        return self._backend.get_slot_number(self._node_index, self._attr_type)

    def on_value_changed(self, cb):
        """Set the value changed callback."""
        self._cb = cb


cdef class NodeBase:
    @property
    def index(self):
        """int: Index of current node instance."""
        return self._index

    @property
    def is_deleted(self):
        """bool:: Is this node instance already been deleted."""
        return self._is_deleted

    cdef void setup(self, BackendAbc backend, NODE_INDEX index, NODE_TYPE node_type, dict attr_name_type_dict) except *:
        """Setup frame node, and bind attributes."""
        self._index = index
        self._type = node_type
        self._backend = backend
        self._is_deleted = False
        self._attributes = attr_name_type_dict

        self._bind_attributes()

    cdef void _bind_attributes(self) except *:
        """Bind attributes declared in class."""
        cdef dict __dict__ = object.__getattribute__(self, "__dict__")

        cdef ATTR_TYPE attr_type
        cdef str cb_name
        cdef _NodeAttributeAccessor attr_acc

        for name in dir(type(self)):
            attr = getattr(self, name)

            # Append an attribute access wrapper to current instance.
            if isinstance(attr, NodeAttribute):
                # Register attribute.
                attr_type = self._attributes[name]
                attr_acc = _NodeAttributeAccessor(attr, attr_type, self._backend, self._index)

                # NOTE: Here we have to use __dict__ to avoid infinite loop, as we override __getattribute__
                # NOTE: we use attribute name here to support get attribute value by name from python side.
                __dict__[name] = attr_acc

                # Bind a value changed callback if available, named as _on_<attr name>_changed.
                # Except list attribute.
                # if not attr_acc._is_list:
                cb_name = f"_on_{name}_changed"
                cb_func = getattr(self, cb_name, None)

                if cb_func is not None:
                    attr_acc.on_value_changed(cb_func)

    def __setattr__(self, name, value):
        """Used to avoid attribute overriding, and an easy way to set for 1 slot attribute."""
        if self._is_deleted:
            raise BackendsAccessDeletedNodeException()

        cdef dict __dict__ = self.__dict__
        cdef str attr_name = name

        if attr_name in __dict__:
            attr_acc = __dict__[attr_name]

            if isinstance(attr_acc, _NodeAttributeAccessor):
                if not attr_acc._is_list and attr_acc._slot_number > 1:
                    raise BackendsArrayAttributeAccessException()
                else:
                    # Short-hand for attributes with 1 slot.
                    attr_acc[0] = value
            else:
                __dict__[attr_name] = value
        else:
            __dict__[attr_name] = value

    def __getattribute__(self, name):
        """Provide easy way to get attribute with 1 slot."""
        cdef dict __dict__ = self.__dict__
        cdef str attr_name = name

        if attr_name in __dict__:
            attr_acc = __dict__[attr_name]

            if isinstance(attr_acc, _NodeAttributeAccessor):
                if self._is_deleted:
                    raise BackendsAccessDeletedNodeException()

                # For list attribute, we do not support ignore index.
                if not attr_acc._is_list and attr_acc._slot_number == 1:
                    return attr_acc[0]

            return attr_acc

        return super().__getattribute__(attr_name)


cdef class FrameNode:
    def __cinit__(self, type node_cls, NODE_INDEX number):
        self._node_cls = node_cls
        self._number = number


cdef class FrameBase:
    def __init__(self, enable_snapshot: bool = False, total_snapshot: int = 0, options: dict = {}, backend_name=None):
        # Backend name from parameter has highest priority.
        if backend_name is None:
            # Try to get default backend settings from environment settings, or use default.
            backend_name = os.environ.get("DEFAULT_BACKEND_NAME", _default_backend_name)

        backend = backend_dict.get(backend_name, NumpyBackend)

        self._backend_name = "static" if backend == NumpyBackend else "dynamic"

        self._backend = backend()

        self._node_cls_dict = {}
        self._node_origin_number_dict = {}
        self._node_name2attrname_dict = {}

        self._setup_backend(enable_snapshot, total_snapshot, options)

    @property
    def backend_type(self) -> str:
        """str: Type of backend, static or dynamic."""
        return self._backend_name

    @property
    def snapshots(self) -> SnapshotList:
        """SnapshotList: Snapshots of this frame."""
        return self._snapshot_list

    def get_node_info(self) -> dict:
        """Get a dictionary contains node attribute and number definition.

        Returns:
            dict: Key is node name in Frame, value is a dictionary contains attribute and number.
        """
        return self._backend.get_node_info()

    cpdef void reset(self) except *:
        """Reset internal states of frame, currently all the attributes will reset to 0.

        Note:
            This method will not reset states in snapshot list.
        """
        self._backend.reset()

        cdef NodeBase node

        if self._backend.is_support_dynamic_features():
            # We need to make sure node number same as origin after reset.
            for node_name, node_number in self._node_origin_number_dict.items():
                node_list = self.__dict__[self._node_name2attrname_dict[node_name]]

                for i in range(len(node_list)-1, -1, -1):
                    node = node_list[i]

                    if i >= node_number:
                        del node_list[i]
                    else:
                        node._is_deleted = False

            # Also

    cpdef void take_snapshot(self, INT tick) except *:
        """Take snapshot for specified point (tick) for current frame.

        This method will copy current frame value (except const attributes) into snapshot list for later using.

        NOTE:
            Frame and SnapshotList do not know about snapshot_resolution from simulator,
            they just accept a point as tick for current frame states.
            Current scenarios and decision event already provided a property "frame_index"
            used to get correct point of snapshot.

        Args:
            tick (int): Tick (point or frame index) for current frame states,
                this value will be used when querying states from snapshot list.
        """
        if self._backend.snapshots is not None:
            self._backend.snapshots.take_snapshot(tick)

    cpdef void enable_history(self, str path) except *:
        """Enable snapshot history, history will be dumped into files under specified folder,
        history of nodes will be dump seperately, named as node name.

        Different with take snapshot, history will not over-write oldest or snapshot at same point,
        it will keep all the changes after ``take_snapshot`` method is called.

        Args:
            path (str): Folder path to save history files.
        """
        if self._backend.snapshots is not None:
            self._backend.snapshots.enable_history(path)

    cpdef void append_node(self, str node_name, NODE_INDEX number) except +:
        """Append specified number of node instance to node type.

        Args:
            node_name (str): Name of the node type to append.
            number (int): Number of node instance to append.
        """
        cdef NODE_TYPE node_type
        cdef NodeBase node
        cdef NodeBase first_node
        cdef list node_list

        if self._backend.is_support_dynamic_features() and number > 0:
            node_list = self.__dict__.get(self._node_name2attrname_dict[node_name], None)

            if node_list is None:
                raise BackendsInvalidNodeException()

            # Get the node type for furthur using.
            first_node = node_list[0]
            node_type = first_node._type

            self._backend.append_node(node_type, number)

            # Append instance to list.
            for i in range(number):
                node = self._node_cls_dict[node_name]()

                node.setup(self._backend, len(node_list), node_type, first_node._attributes)

                node_list.append(node)

    cpdef void delete_node(self, NodeBase node) except +:
        """Delete specified node instance, then any operation on this instance will cause error.

        Args:
            node (NodeBase): Node instance to delete.
        """
        if self._backend.is_support_dynamic_features():
            self._backend.delete_node(node._type, node._index)

            node._is_deleted = True

    cpdef void resume_node(self, NodeBase node) except +:
        """Resume a deleted node instance, this will enable operations on this node instance.

        Args:
            node (NodeBase): Node instance to resume.
        """
        if self._backend.is_support_dynamic_features() and node._is_deleted:
            self._backend.resume_node(node._type, node._index)

            node._is_deleted = False

    def dump(self, folder: str):
        """Dump data of current frame into specified folder.

        Args:
            folder (str): Folder path to dump (without file name).
        """
        if os.path.exists(folder):
            self._backend.dump(folder)

    cdef void _setup_backend(self, bool enable_snapshot, USHORT total_snapshots, dict options) except *:
        """Setup Frame for further using."""
        cdef str frame_attr_name
        cdef str node_attr_name
        cdef str node_name
        cdef NODE_TYPE node_type
        cdef ATTR_TYPE attr_type
        cdef type node_cls

        cdef list node_instance_list

        # Attr name -> type.
        cdef dict attr_name_type_dict = {}

        # Node -> attr -> type.
        cdef dict node_attr_type_dict = {}
        cdef dict node_type_dict = {}
        cdef NODE_INDEX node_number
        cdef NodeBase node

        # Internal loop indexer.
        cdef NODE_INDEX i

        # Register node and attribute in backend.
        for frame_attr_name in dir(type(self)):
            frame_attr = getattr(self, frame_attr_name)

            # We only care about FrameNode instance.
            if isinstance(frame_attr, FrameNode):
                node_cls = frame_attr._node_cls
                node_number = frame_attr._number
                node_name = node_cls.__node_name__

                self._node_cls_dict[node_name] = node_cls
                self._node_origin_number_dict[node_name] = node_number

                # Temp list to hold current node instances.
                node_instance_list = [None] * node_number

                # Register node.
                node_type = self._backend.add_node(node_name, node_number)

                # Used to collect node type and its name, then initial snapshot list with different nodes.
                node_type_dict[node_name] = node_type

                # Used to collect attributes for current node, then initial node instance with it.
                attr_name_type_dict = {}

                # Register attributes.
                for node_attr_name in dir(node_cls):
                    node_attr = getattr(node_cls, node_attr_name)

                    if node_attr and isinstance(node_attr, NodeAttribute):
                        attr_type = self._backend.add_attr(node_type, node_attr_name, node_attr._dtype, node_attr._slot_number, node_attr._is_const, node_attr._is_list)

                        attr_name_type_dict[node_attr_name] = attr_type

                node_attr_type_dict[node_name] = attr_name_type_dict

                # Create instance.
                for i in range(node_number):
                    node = node_cls()

                    # Setup each node instance.
                    node.setup(self._backend, i, node_type, attr_name_type_dict)

                    node_instance_list[i] = node

                # Make it possible to get node instance list by their's name.
                self.__dict__[frame_attr_name] = node_instance_list
                self._node_name2attrname_dict[node_name] = frame_attr_name

        # Setup backend to allocate memory.
        self._backend.setup(enable_snapshot, total_snapshots, options)

        if enable_snapshot:
            self._snapshot_list = SnapshotList(node_type_dict, node_attr_type_dict, self._backend.snapshots)

    def dump(self, filePath):
        self._backend.dump(filePath)

# Wrapper to access specified node in snapshots (read-only), to provide quick way for querying.
# All the slice interface will start from here to construct final parameters.
cdef class SnapshotNode:
    cdef:
        # Target node id.
        NODE_TYPE _node_type

        # Attributes: name -> id.
        dict _attributes

        # Reference to snapshots for querying.
        SnapshotListAbc _snapshots

    def __cinit__(self, NODE_TYPE node_type, dict attributes, SnapshotListAbc snapshots):
        self._node_type = node_type
        self._snapshots = snapshots
        self._attributes = attributes

    def __len__(self):
        """Number of current node."""
        return self._snapshots.get_node_number(self._node_type)

    def __getitem__(self, key: slice):
        """Used to support states slice querying."""

        cdef list ticks = []
        cdef list node_list = []
        cdef list attr_list = []

        cdef type start_type = type(key.start)
        cdef type stop_type = type(key.stop)
        cdef type step_type = type(key.step)

        # Prepare ticks.
        if key.start is None:
            ticks = []
        elif start_type is tuple or start_type is list:
            ticks = list(key.start)
        else:
            ticks.append(key.start)

        # Prepare node index list.
        if key.stop is None:
            node_list = []
        elif stop_type is tuple or stop_type is list:
            node_list = list(key.stop)
        else:
            node_list.append(key.stop)

        # Querying need at least one attribute.
        if key.step is None:
            return None

        # Prepare attribute names.
        if step_type is tuple or step_type is list:
            attr_list = list(key.step)
        else:
            attr_list = [key.step]

        cdef str attr_name
        cdef list attr_type_list = []

        # Make sure all attributes exist.
        for attr_name in attr_list:
            if attr_name not in self._attributes:
                raise BackendsInvalidAttributeException()

            attr_type_list.append(self._attributes[attr_name])

        return self._snapshots.query(self._node_type, ticks, node_list, attr_type_list)


cdef class SnapshotList:
    def __cinit__(self, dict node_type_dict, dict node_attr_type_dict, SnapshotListAbc snapshots):
        cdef str node_name
        cdef NODE_TYPE node_type

        self._snapshots = snapshots

        self._nodes_dict = {}

        # Initial for each node type.
        for node_name, node_type in node_type_dict.items():
            self._nodes_dict[node_name] = SnapshotNode(node_type, node_attr_type_dict[node_name], snapshots)

    def get_frame_index_list(self)->list:
        """Get list of available frame index in snapshot list.

        Returns:
            List[int]: Frame index list.
        """
        return self._snapshots.get_frame_index_list()

    def __getitem__(self, name: str):
        """Used to get slice querying interface for specified node."""
        cdef str node_name = name

        return self._nodes_dict.get(node_name, None)

    def __len__(self):
        """Max size of snapshot."""
        return len(self._snapshots)

    def reset(self):
        """Reset current states, this will cause all the values to be 0, make sure call it after states querying."""
        self._snapshots.reset()

    def dump(self, folder: str):
        """Dump data of current snapshots into specified folder.

        Args:
            folder (str): Folder path to dump (without file name).
        """
        if os.path.exists(folder):
            self._snapshots.dump(folder)
