# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport cython
from cpython cimport bool
from typing import Union

from maro.utils.exception.backends_exception import BackendsGetItemInvalidException, BackendsSetItemInvalidException, BackendsArrayAttributeAccessException
from maro.backends.backend cimport BackendAbc, SnapshotListAbc

# NOTE: here to support backend switching
IF FRAME_BACKEND == "NUMPY":
    cimport numpy as np
    import numpy as np

    from maro.backends.np_backend cimport NumpyBackend as backend

ELSE:
    from maro.backends.raw_backend cimport RawBackend as backend


def node(name: str):
    """Frame node decorator, used to specified node name in Frame and SnapshotList.
    
    This node name used for querying from snapshot list, see NodeBase for details.

    Args:
        name(str): node name in Frame
    """
    def node_dec(cls):
        cls.__node_name__ = name

        return cls

    return node_dec


cdef class NodeAttribute:
    def __cinit__(self, dtype: str, slot_num: int = 1):
        self.dtype = dtype
        self.slot_number = slot_num


# TODO: a better way to support multiple value get/set for an attribute with more than one slot
# 
# Wrapper to provide easy way to access attribute value of specified node
# with this wrapper, user can get/set attribute value more easily
cdef class NodeAttributeAccessor:
    cdef:
        # target node
        str _node_name
        # attribute name
        str _attr_name
        BackendAbc _backend
        int _index

        public NodeAttribute attr

        # enable dynamic attributes
        dict __dict__

        # slot list cache, used to avoid to much runtime list generation
        # slot -> int[:]
        dict _slot_list_cache
    
    def __init__(self, attr: NodeAttribute, node_name: str, attr_name: str, backend: BackendAbc, index: int):
        self.attr = attr
        self._node_name = node_name
        self._attr_name = attr_name
        self._backend = backend
        self._index = index
        self._slot_list_cache = {}

    def __getitem__(self, slot: Union[int, slice, list, tuple]):
        """We use this function to support slice interface to access attribute, like:

            a = node.attr1[1:]
            b = node.attr1[:]
            c = node.attr1[(1, 2, 3)]
        """

        cdef int start
        cdef int stop
        cdef type slot_type = type(slot)
        cdef int[:] slot_list

        # node.attribute[0]
        if slot_type == int:
            return self._backend.get_attr_value(self._node_name, self._index, self._attr_name, slot)

        # try to support following:
        # node.attribute[1:3] 
        # node.attribute[[1, 2, 3]]
        # node.attribute[(0, 1)]
        cdef tuple slot_key = tuple(slot) if slot_type != slice else (slot.start, slot.stop, slot.step)

        slot_list = self._slot_list_cache.get(slot_key, None)

        if slot_list is None:
            if slot_type == slice:
                start = 0 if slot.start is None else slot.start
                stop = self.attr.slot_number if slot.stop is None else slot.stop

                slot_list = np.arange(start, stop, dtype='i')
            elif slot_type == list or slot_type == tuple:
                slot_list = np.array(slot, dtype='i')
            else:
                raise BackendsGetItemInvalidException()

            self._slot_list_cache[slot_key] = slot_list

        return self._backend.get_attr_values(self._node_name, self._index, self._attr_name, slot_list)

    def __setitem__(self, slot: Union[int, slice, list, tuple], value: Union[object, list, tuple, np.ndarray]):
        # check if type match
        cdef int[:] slot_list
        cdef list values

        # TODO: use large data type for index
        cdef int start
        cdef int stop

        cdef type slot_type = type(slot)
        cdef type value_type = type(value)
        
        cdef int values_length
        cdef int slot_length
        cdef tuple slot_key

        # node.attribute[0] = 1
        if slot_type == int:
            self._backend.set_attr_value(self._node_name, self._index, self._attr_name, slot, value)
        elif slot_type == list or slot_type == tuple or slot_type == slice:
            # try to support following:
            # node.attribute[0: 2] = 1/[1,2]/ (0, 2, 3)
            slot_key = tuple(slot) if slot_type != slice else (slot.start, slot.stop, slot.step)

            slot_list = self._slot_list_cache.get(slot_key, None)

            if slot_list is None:
                if slot_type == slice:
                    start = 0 if slot.start is None else slot.start
                    stop = self.attr.slot_number if slot.stop is None else slot.stop

                    slot_list = np.arange(start, stop, dtype='i')
                elif slot_type == list or slot_type == tuple:
                    slot_list = np.array(slot, dtype='i')

                self._slot_list_cache[slot_key] = slot_list
            
            slot_length = len(slot_list)

            if value_type == list or value_type == tuple or value_type == np.ndarray:
                values = list(value)

                values_length = len(values)

                # make sure the value size is same as slot size
                if values_length > slot_length:
                    values = values[0: slot_length]
                elif values_length < slot_length:
                    slot_list = slot_list[0: values_length]
            else:
                values = [value] * slot_length

            self._backend.set_attr_values(self._node_name, self._index, self._attr_name, slot_list, values)
        else:
            raise BackendsSetItemInvalidException()

        # check and invoke value changed callback
        if "_cb" in self.__dict__:
            self._cb(value)

    def on_value_changed(self, cb):
        """Set the value changed callback"""
        self._cb = cb
        

cdef class NodeBase:
    @property
    def index(self):
        return self._index

    cdef void setup(self, BackendAbc backend, int index) except *:
        """setup frame node, and bind attributes"""
        self._index = index
        self._backend = backend

        self._bind_attributes()

    cdef void _bind_attributes(self) except *:
        """bind attributes declared in class"""
        cdef dict __dict__ = object.__getattribute__(self, "__dict__")    

        cdef str name
        cdef str node_name
        cdef str cb_name
        cdef NodeAttributeAccessor attr_acc

        for name, attr in type(self).__dict__.items():
            # append an attribute access wrapper to current instance
            if isinstance(attr, NodeAttribute):
                # TODO: this will override exist attribute of sub-class instance, maybe a warning later
                node_name = getattr(type(self), "__node_name__", None)

                # NOTE: here we have to use __dict__ to avoid infinite loop, as we override __getattribute__
                attr_acc = NodeAttributeAccessor(attr, node_name, name, self._backend, self._index)

                __dict__[name] = attr_acc

                # bind a value changed callback if available, named as _on_<attr name>_changed
                cb_name = f"_on_{name}_changed"
                cb_func = getattr(self, cb_name, None)

                if cb_func is not None:    
                    attr_acc.on_value_changed(cb_func)

    def __setattr__(self, name, value):
        """Used to avoid attribute overriding, and an easy way to set for 1 slot attribute"""
        cdef dict __dict__ = self.__dict__ 
        cdef str attr_name = name
     
        if attr_name in __dict__:
            attr_acc = __dict__[attr_name]

            if isinstance(attr_acc, NodeAttributeAccessor):
                if attr_acc.attr.slot_number > 1:
                    raise BackendsArrayAttributeAccessException()
                else:
                    # short-hand for attributes with 1 slot
                    attr_acc[0] = value
            else:
                __dict__[attr_name] = value
        else:
            __dict__[attr_name] = value

    def __getattribute__(self, name):
        """provide easy way to get attribute with 1 slot"""
        cdef dict __dict__ = self.__dict__     
        cdef str attr_name = name 

        if attr_name in __dict__:
            attr_acc = __dict__[attr_name]

            if isinstance(attr_acc, NodeAttributeAccessor):
                if attr_acc.attr.slot_number == 1:
                    return attr_acc[0]

            return attr_acc
          
        return super().__getattribute__(attr_name)


cdef class FrameNode:
    def __cinit__(self, type node_cls, int number):
        self.node_cls = node_cls
        self.number = number


cdef class FrameBase:
    def __init__(self, enable_snapshot: bool = False, total_snapshot: int = 0, options: dict = {}):
        self._backend = backend()
        
        self._setup_backend(enable_snapshot, total_snapshot, options)

    @property
    def snapshots(self) -> SnapshotList:
        """Snapshots of this frame"""
        return self._snapshot_list

    def get_node_info(self) -> dict:
        """Get node information.
        
        Returns:
            dict: key is node name in Frame, key is a dictionary contains attribute and number 
        """
        return self._backend.get_node_info()

    cpdef void reset(self) except *:
        self._backend.reset()

    cpdef void take_snapshot(self, int tick) except *:
        if self._backend.snapshots is not None:
            self._backend.snapshots.take_snapshot(tick)

    cpdef void enable_history(self, str path) except *:
        if self._backend.snapshots is not None:
            self._backend.snapshots.enable_history(path)

    cdef void _setup_backend(self, bool enable_snapshot, int total_snapshots, dict options) except *:
        """setup Frame for further using"""
        cdef str frame_attr_name
        cdef str node_attr_name
        cdef str node_name
        cdef type node_cls
        
        cdef list node_instance_list
        cdef dict node_name_num_dict = {} # node name -> node number dict
        cdef int node_number
        cdef NodeBase node

        # internal loop indexer
        cdef int i

        cdef list node_def_list = []
        
        # register node and attribute in backend
        #for node_cls in self._node_def_list:
        for frame_attr_name, frame_attr in type(self).__dict__.items():
            if isinstance(frame_attr, FrameNode):
                node_cls = frame_attr.node_cls
                node_number = frame_attr.number
                node_name = node_cls.__node_name__

                # temp list to hold current node instances
                node_instance_list = [None] * node_number

                node_name_num_dict[node_name] = node_number

                # register node
                self._backend.add_node(node_name, node_number)

                # register attribute
                for node_attr_name, node_attr in node_cls.__dict__.items():
                    if isinstance(node_attr, NodeAttribute):
                        self._backend.add_attr(node_name, node_attr_name, node_attr.dtype, node_attr.slot_number)

                # create instance
                for i in range(node_number):
                    node = node_cls()

                    # pass the backend reference and index
                    node.setup(self._backend, i)

                    node_instance_list[i] = node

                # add dynamic fields
                self.__dict__[frame_attr_name] = node_instance_list
        
        # setup backend to allocate memory
        self._backend.setup(enable_snapshot, total_snapshots, options)

        if enable_snapshot:
            self._snapshot_list = SnapshotList(node_name_num_dict, self._backend.snapshots)


# Wrapper to access specified node in snapshots (read-only), to provide quick way for querying.
# All the slice interface will start from here to construct final parameters
cdef class SnapshotNode:
    cdef:
        # target node number, used for empty node list
        int _node_number

        # target node name
        str _node_name
        
        # reference to snapshots for querying
        SnapshotListAbc _snapshots

    def __cinit__(self, str node_name, int node_number, SnapshotListAbc snapshots):
        self._node_name = node_name
        self._node_number = node_number
        self._snapshots = snapshots

    def __len__(self):
        """Number of current node"""
        return self._node_number

    def __getitem__(self, key: slice):
        """Used to support states slice querying"""

        cdef list ticks = []
        cdef list node_list = []
        cdef list attr_list = []

        cdef type start_type = type(key.start)
        cdef type stop_type = type(key.stop)
        cdef type step_type = type(key.step)

        # ticks
        if key.start is None:
            ticks = []
        elif start_type is tuple or start_type is list:
            ticks = list(key.start)
        else:
            ticks.append(key.start)
        
        # node id list
        if key.stop is None:
            node_list = []
        elif stop_type is tuple or stop_type is list:
            node_list = list(key.stop)
        else:
            node_list.append(key.stop)

        if key.step is None:
            return None
        
        # attribute names
        if step_type is tuple or step_type is list:
            attr_list = list(key.step)
        else:
            attr_list = [key.step]    

        return self._snapshots.query(self._node_name, ticks, node_list, attr_list)    


cdef class SnapshotList:
    def __cinit__(self, dict node_name_num_dict, SnapshotListAbc snapshots):
        cdef str node_name
        cdef int node_number
        
        self._snapshots = snapshots

        self._nodes_dict = {}

        for node_name, node_number in node_name_num_dict.items():
            self._nodes_dict[node_name] = SnapshotNode(node_name, node_number, snapshots)

    def get_frame_index_list(self)->list:
        """Get list of available frame index in snapshot list
        
        Returns:
            List[int]: frame index list
        """
        return self._snapshots.get_frame_index_list()

    def __getitem__(self, name: str):
        """used to get slice querying interface for specified node"""
        cdef str node_name = name

        return self._nodes_dict.get(node_name, None)

    def __len__(self):
        """Max size of snapshot"""
        return len(self._snapshots)

    def reset(self):
        """Reset current snapshot"""
        self._snapshots.reset()