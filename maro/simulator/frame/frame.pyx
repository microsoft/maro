#cython: language_level=3

import sys
import os
import numpy as np
cimport numpy as np
cimport cython

from enum import IntEnum, Enum
from cython cimport view
from cpython cimport bool
from math import ceil
from collections.abc import Iterable

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t

class FrameError(Exception):
    '''Base exception of frame'''
    def __init__(self, msg):
        self.message = msg

class FrameInvalidAttributeDataType(FrameError):
    def __init__(self, attr_name, dtype):
        super().__init__(f"Invalid data type {dtype} for attribute {attr_name}, use any numpy.dtype instead")

class FrameAttributeExistError(FrameError):
    '''Try to register attribute with exist name'''
    def __init__(self, attr_name):
        super().__init__(f"Attribute {attr_name} already exist")

class FrameAttributeNotFoundError(FrameError):
    '''Try to access frame with not registered attribute'''
    def __init__(self, attr_name):
        super().__init__(f"Not exist attribute name {attr_name}")

class SnapshotNoAttributeProvide(FrameError):
    def __init__(self):
        super().__init__("Must provide an attribute name to query")

class FrameNodeType(Enum):
    STATIC="static"
    DYNAMIC="dynamic"
    GENERAL="general"

AT_STATIC = FrameNodeType.STATIC
AT_DYNAMIC = FrameNodeType.DYNAMIC
AT_GENERAL = FrameNodeType.GENERAL # used for general purpose


cdef class FrameAttribute:
    '''Used to wrapper attribute accessing information internally'''
    cdef:
        public int32_t slot_num
        public str ntype
        public np.dtype dtype
        public str name

    def __cinit__(self, str ntype, str name, dtype, int32_t slot_num):
        self.ntype = ntype
        self.name = name
        self.slot_num = slot_num

        if type(dtype) is str:
            self.dtype = np.dtype(dtype)
        elif type(dtype) is np.dtype:
            self.dtype = np.dtype
        else:
            raise FrameInvalidAttributeDataType()

        # TODO: exclude un-supported dtype later

    def get_dtype(self):
        """Return a numpy compace structure data type"""

        # use this to avoid future-warning from numpy
        if self.slot_num == 1:
            return (self.name, self.dtype)
        else:
            return (self.name, self.dtype, self.slot_num)

cdef class Frame:
    """Frame used to hold attributes for both static and dynamic nodes.
    To initialize a frame, attributes must to be registered before setup.
    Example:
        Create a simple frame that with 10 static and 10 dynamic nodes (default batch number is 1), and attributes like "attr1", "attr2":
            
            static_node_num = 10
            dynamic_node_num = 10

            # init the frame object first
            frame = Frame(static_node_num, dynamic_node_num)

            # then register attributes
            # register an attribute named "attr1", its data type is float, can hold 1 value (slot)
            frame.register_attribute(FrameNodeType.DYNAMIC, "attr1", FrameDataType.FLOAT, 1)
            
            # register an attribute named "attr2", its data type is int, can hold 2 value (slots)
            frame.register_attribute(FrameNodeType.STATIC, "attr2", FrameDataType.INT32, 2)
            
            # then we can setup the frame for using
            frame.setup()
            
            # the frame is ready to accessing now
            # get an attribute (first slot) of a static node that id is 0
            a1 = frame[FrameNodeType.DYNAMIC, 0, "attr1", 0]
            
            # set an attribute (2nd slot) of a dynamic node that id is 0
            frame[FrameNodeType.DYNAMIC, 0, "attr1", 1] = 123
    Args:
        static_node_num (int): number of static nodes in frame
        dynamic_node_num (int): number of dynamic nodes in frame
    """
    cdef:
        int32_t _static_node_num
        int32_t _dynamic_node_num
        dict _node_num_map

        dict _attr_dict
        dict _data_dict
        
        # used to cache attribute by atype
        dict _grouped_attr_dict

        bool _is_initialized

    def __cinit__(self, static_node_num, dynamic_node_num):
        self._node_num_map = {}
        self._attr_dict = {}
        self._data_dict = {}
        self._grouped_attr_dict = {}

        self._node_num_map[AT_GENERAL] = 1
        self._node_num_map[AT_DYNAMIC] = dynamic_node_num
        self._node_num_map[AT_STATIC] = static_node_num

        self._static_node_num = static_node_num
        self._dynamic_node_num = dynamic_node_num

        self._is_initialized = False

    @property
    def static_node_number(self):
        '''int: Number of static nodes in current frame'''
        return self._static_node_num

    @property
    def dynamic_node_number(self):
        '''int: Number of dynamic nodes in current frame'''
        return self._dynamic_node_num

    def add_node_type(self, name: str):
        self._node_num_map[name] = 0

    def register_attribute(self, ntype: FrameNodeType, name: str, dtype, slot_num=1):
        '''Register an attribute for nodes in frame, then can access the new attribute with get/set_attribute methods.
        NOTE: this method should be called before setup method
        Args:
            ntype (FrameNodeType): type of this attribute belongs to
            name (str): name of the attribute 
            data_type (numpy.dtype): data type of attribute
            slot_num (int): how many slots of this attributes can hold
        Raises:
            FrameAttributeExistError: if the name already being registered
        '''
        attr_key = (ntype, name)

        if attr_key in self._attr_dict:
            raise FrameAttributeExistError(name)

        #NOTE: data layout
        # we can query like arr["attr name"][node_id,slot_index]
        self._attr_dict[attr_key] = FrameAttribute(ntype.value, name, dtype, slot_num)

    def setup(self):
        '''Setup the frame with registered attributes
        '''
        if self._is_initialized:
            return

        cdef list attr_list
        cdef np.dtype t
        
        for ntype in self._node_num_map.keys():
            attr_list = [attr for key, attr in self._attr_dict.items() if key[0] == ntype]

            if len(attr_list) > 0:
                self._grouped_attr_dict[ntype] = attr_list

                t = np.dtype([attr.get_dtype() for attr in attr_list])
                self._data_dict[ntype] = np.zeros(self._node_num_map[ntype], dtype=t)

        self._is_initialized = True

    cpdef reset(self):
        '''Reset all the attributes to default value'''
        for ntype, arr in self._data_dict.items():
            # we have to reset by each attribute
            for attr in self._grouped_attr_dict[ntype]:
                arr[attr.name] = 0

    def __getitem__(self, tuple key):
        '''Get specified attribute value with general way
        Args:
            a tuple with following items:
            atype (FrameNodeType): type of attribute belongs to
            node_id (int): id the the resource node
            attr_name (str): name of accessing attribute
            slot_index (int, optional): index of the attribute slot
        Returns:
            value of specified attribute slot
        '''
        
        ntype = key[0]
        cdef int32_t node_id = key[1]
        cdef str attr_name = key[2]
        cdef int32_t slot_index = 0 if len(key) < 4 else key[3]

        cdef np.ndarray attr_array = self._data_dict[ntype][attr_name]
        cdef FrameAttribute attr = self._attr_dict[(ntype, attr_name)]

        if attr.slot_num > 1:
            return attr_array[node_id, slot_index]
        else:
            return attr_array[node_id]

    def __setitem__(self, tuple key, val):
        '''Set specified attribute value
        Args:
            a tuple with following items:
            atype (FrameNodeType): type of attribute belongs to
            node_id (int): id the the resource node
            attr_name (str): name of accessing attribute
            slot_index (int): index of the attribute slot        
            value (float/int): value to set
        Raises:
            GraphAttributeNotFoundError: specified attribute is not registered
        '''
        atype = key[0]
        cdef int32_t node_id = key[1]
        cdef str attr_name = key[2]
        cdef int32_t slot_index = 0 if len(key) < 4 else key[3]
        cdef int32_t node_num = self._node_num_map[atype]

        cdef np.ndarray attr_array = self._data_dict[atype][attr_name]
        cdef FrameAttribute attr = self._attr_dict[(atype, attr_name)]

        if attr.slot_num > 1:
            attr_array[node_id, slot_index] = val
        else:
            attr_array[node_id] = val


cdef class SnapshotList:
    '''SnapshotList used to hold list of snapshots that taken from Graph object at a certain tick.
    SnapshotList only provide interface to get data, cannot set data.
    Examples:
        it is recommended to use slice to query attributes.
        . snapshot_list.static_nodes[[tick list]: [node id list]: [(attribute name, slot index), ...]]
        . snapshot_list.dynamic_nodes[[tick list]: [node id list]: ((attribute name, slot index), ...)]
        
        all the list parameter can be a single value.
        if tick or node id list is None, then means query all.
        # query 1st slot value of attribute "a1" for node 1 at all the ticks
        snapshot_list.static_nodes[: 1: ("a1", 0)]
        
        # 0 is default slot index, so above query can re-write to
        snapshot_list.static_nodes[: 1: "a1"]
        
        # query 1st and 2nd slot value of attribute "a1" for all the nodes at tick 0
        snapshot_list.static_nodes[0: : ("a1", [0, 1])]
        
        # query 1st slot value for attribute "a1" and "a2" for all the nodes at all the ticks
        snapshot_list.static_nodes[:: ["a1", "a2"]]
        
        # query a matrix at tick 0
        snapshot_list.matrix[0: "m1"]
        
        # query matrix at tick 0 and 1
        snapshot_list.matrix[[0, 1]: "m1"]
    '''
    cdef:
        Frame _frame
        int32_t _max_ticks
        dict _node_num_map

        dict _data_dict
        dict _attr_dict
        dict _grouped_attr_dict
        bool _enable_memmap
        
        SnapshotNodeAccessor _static_node_acc
        SnapshotNodeAccessor _dynamic_node_acc
        SnapshotGeneralAccessor _general_acc

    def __cinit__(self, Frame frame, int32_t max_ticks, enable_memmap=False):
        self._frame = frame
        self._max_ticks = max_ticks
        self._enable_memmap = enable_memmap
        self._data_dict = {}
        self._attr_dict = frame._attr_dict
        self._grouped_attr_dict = frame._grouped_attr_dict
        self._node_num_map = frame._node_num_map

        cdef np.dtype t
        cdef attr_list

        if self._enable_memmap:
            file_path = "data.bin"

            if not os.path.exists(file_path):
                fp = open(file_path, "w")
                fp.close()

        cdef int offset = 0
        cdef int node_num = 0
        cdef np.ndarray tmp_arr

        for ntype, attr_list in self._grouped_attr_dict.items():
            t = np.dtype([attr.get_dtype() for attr in attr_list])
            node_num = self._node_num_map[ntype]

            # NOTE: batch first 2d array, so our query can be much simple and fast
            if self._enable_memmap:
                tmp_arr = np.memmap(file_path, dtype=t, mode="r+", shape=(max_ticks, node_num), offset=offset)
                self._data_dict[ntype] = tmp_arr

                offset += tmp_arr.itemsize * tmp_arr.size
            else:
                self._data_dict[ntype] = np.zeros((max_ticks, node_num), dtype=t)
            
        if AT_STATIC in self._data_dict:
            self._static_node_acc = SnapshotNodeAccessor(self, AT_STATIC)

        if AT_DYNAMIC in self._data_dict:
            self._dynamic_node_acc = SnapshotNodeAccessor(self, AT_DYNAMIC)

        if AT_GENERAL in self._data_dict:
            self._general_acc = SnapshotGeneralAccessor(self)

    @property
    def dynamic_node_number(self):
        '''int: Dynamic node number in each snapshot'''
        return self._graph.dynamic_node_number

    @property
    def static_node_number(self):
        '''int: Static node number in each snapshot'''
        return self._graph.static_node_number

    @property
    def matrix(self):
        return self._general_acc

    @property
    def general_nodes(self):
        return self._general_acc

    @property
    def static_nodes(self):
        '''Same as dynamic_nodes, but for static nodes'''

        return self._static_node_acc
    
    @property
    def dynamic_nodes(self):
        '''Slice interface to query attribute value of dynamic nodes.

        The slice like [tick: id: (attribute name, slot index]

        tick: tick to query, can be a list
        id: id of dynamic nodes to query, can be a list
        attribute name: registered attribute to query, can be a list
        slot index: slot to query, can be a list

        Examples:
            # query value of attribute "a1" for node 1 at all the ticks
            snapshot_list.dynamic_nodes[: 1: "a1"]

            # query value of attribute "a1" for all the nodes at tick 0
            snapshot_list.dynamic_nodes[0: : "a"]

            # query 1st slot value for attribute "a1" and "a2" for all the nodes at all the ticks
            snapshot_list.dynamic_nodes[:: ["a1", "a2"]]

        Returns:
            np.ndarray: states numpy array (1d)
        '''
        return self._dynamic_node_acc

    @property
    def attributes(self):
        '''List of the attributes information in current snapshot
        Returns:
            list: A list of attribute details
        '''
        result = []

        for _, attr in self._attr_dict:
            result.append({
                "name": attr.name,
                "slot length": attr.slot_num,
                "attribute type": attr.atype
            })

        return result    

    cpdef insert_snapshot(self, int32_t tick):
        '''Insert a snapshot from graph'''
        
        for ntype, arr in self._data_dict.items():
            arr[tick] = self._frame._data_dict[ntype]

    def reset(self):
        """Reset snapshot list
        """
        for atype, arr in self._data_dict.items():
            for attr in self._grouped_attr_dict[atype]:
                arr[attr.name] = 0

    def __len__(self):
        return self._max_ticks

cdef class SnapshotNodeAccessor:
    """
    Wrapper to access node attributes with slice interface
    """
    cdef:
        int32_t _node_num
        int32_t _max_ticks

        np.ndarray _data_arr

        list _all_ticks
        list _all_nodes
        dict _attr_dict

    def __cinit__(self, SnapshotList snapshots, ntype: FrameNodeType):
        self._node_num = snapshots._node_num_map[ntype] # snapshots._frame.static_node_num if ntype == AT_STATIC else snapshots._frame.dynamic_node_num
        self._max_ticks = snapshots._max_ticks
        self._data_arr = snapshots._data_dict[ntype]
        self._attr_dict = {}
        
        cdef FrameAttribute attr

        for attr in snapshots._grouped_attr_dict[ntype]:
            self._attr_dict[attr.name] = attr

        self._all_ticks = [i for i in range(self._max_ticks)]
        self._all_nodes = [i for i in range(self._node_num)]

    def __len__(self):
        return self._node_num

    def __getitem__(self, slice key):
        cdef list ticks = []
        cdef list node_list = []
        cdef list attr_list = []
        cdef int32_t tick, i, nid
        cdef list retq = []
        cdef int32_t slot_index
        cdef str aname
        cdef list slot_list = []
        cdef list sindex
        cdef FrameAttribute attr

        # ticks
        if key.start is None:
            ticks = self._all_ticks
        elif type(key.start) is tuple or type(key.start) is list:
            ticks = list(key.start)
        else:
            ticks.append(key.start)

        # node id list
        if key.stop is None:
            node_list = self._all_nodes
        elif type(key.stop) is tuple or type(key.start) is list:
            node_list = list(key.stop)
        else:
            node_list.append(key.stop)

        if key.step is None:
            return None
        
        # attribute names
        if type(key.step) is tuple or type(key.step) is list:
            attr_list = list(key.step)
        else:
            attr_list = [key.step]

        # filter ticks
        for tick in ticks:
            for aname in attr_list:
                if aname not in self._attr_dict:
                    raise FrameAttributeNotFoundError(aname)

                if 0 <= tick < self._max_ticks:
                    retq.append(self._data_arr[aname][tick, node_list].astype("f").flatten())
                else:
                    attr = self._attr_dict[aname]
                    retq.append(np.zeros(len(node_list) * attr.slot_num, dtype='f'))

        return np.concatenate(retq)


cdef class SnapshotGeneralAccessor:
    """
    Wrapper to access general data with slice interface
    """
    cdef:
        np.ndarray arr
        list all_ticks
        int32_t max_ticks
        dict attr_dict

    def __cinit__(self, SnapshotList ss):
        self.arr = ss._data_dict[AT_GENERAL]
        self.attr_dict = ss._attr_dict
        self.max_ticks = ss._max_ticks
        self.all_ticks = [i for i in range(self.max_ticks)]


    def __getitem__(self, slice item):
        cdef list ticks
        cdef str attr_name = item.stop
        cdef int32_t tick
        cdef list retq = []
        cdef list rows
        cdef FrameAttribute attr

        key = (AT_GENERAL, attr_name)

        if key not in self.attr_dict:
            return []

        attr = self.attr_dict[key]

        if item.start is None:
            ticks = self.all_ticks
        if type(item.start) is not list:
            ticks = [item.start]
        else:
            ticks = item.start

        for tick in ticks:
            if tick < self.max_ticks:
                retq.append(self.arr[attr_name][:, tick])
            else:
                retq.append(np.zeros(attr.slot_num, dtype='f'))

        return np.concatenate(retq, axis=1)