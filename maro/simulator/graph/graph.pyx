#cython: language_level=3

import sys
import numpy as np
cimport numpy as np
cimport cython

from enum import IntEnum
from cython cimport view
from cpython cimport bool
from math import ceil

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.stdio cimport fopen, fwrite, fread, fclose, FILE

# fused type for graph data
ctypedef fused graph_data_type:
    int8_t
    int16_t
    int32_t
    int64_t
    float

class GraphDataType(IntEnum):
    """
    Data type of graph
    """
    BYTE = 0
    SHORT = 1
    INT32 = 2
    INT64 = 3
    FLOAT = 4
    
class GraphAttributeType(IntEnum):
    """
    Type of attribute belongs to
    """
    STATIC_NODE = 0
    DYNAMIC_NODE = 1
    GENERAL = 2

# for internal use
cdef int8_t DT_BYTE = GraphDataType.BYTE
cdef int8_t DT_SHORT = GraphDataType.SHORT
cdef int8_t DT_INT32 = GraphDataType.INT32
cdef int8_t DT_INT64 = GraphDataType.INT64
cdef int8_t DT_FLOAT = GraphDataType.FLOAT

cdef int8_t AT_STATIC = GraphAttributeType.STATIC_NODE
cdef int8_t AT_DYNAMIC = GraphAttributeType.DYNAMIC_NODE
cdef int8_t AT_GENERAL = GraphAttributeType.GENERAL

# mapping from data type to size
# TODO: any good ways for this?
cdef dict dtype_size_map = {
    DT_BYTE : sizeof(int8_t),
    DT_SHORT : sizeof(int16_t),
    DT_INT32 : sizeof(int32_t),
    DT_INT64 : sizeof(int64_t),
    DT_FLOAT : sizeof(float),
}


class GraphError(Exception):
    '''Base exception of graph'''
    def __init__(self, msg):
        self.message = msg


class GraphAttributeNotFoundError(GraphError):
    '''Try to access graph with not registered attribute'''
    def __init__(self, msg):
        super().__init__(msg)


class GraphAttributeExistError(GraphError):
    '''Try to register attribute with exist name'''
    def __init__(self, msg):
        super().__init__(msg)


class GraphAttributeNotRegisteredError(GraphError):
    '''Try to setup a graph without registered any attributes'''
    def __init__(self):
        super().__init__("Graph has no attributes registered.")


class GraphAlreadySetupError(GraphError):
    '''Try to register an attribute after setup'''
    def __init__(self):
        super().__init__("Graph already being setup, cannot register attributes.")


class GraphAlreadySetupError(GraphError):
    '''Try to register an attribute after setup'''
    def __init__(self):
        super().__init__("Graph already being setup, cannot register attributes.")


class SnapshotAccessError(GraphError):
    '''Snapshot cannot be wrote'''
    def __init__(self):
        super().__init__("Snapshot cannot be wrote.")


class SnapshotInvalidTick(GraphError):
    """Using invalid parameter to take snapshot"""
    def __init__(self, msg):
        super().__init__(msg)


cdef class GraphAttribute:
    '''Used to wrapper attribute accessing information internally'''
    cdef:
        # data type: GraphDataType
        public int8_t dtype

        # attribute type
        public int8_t atype

        # size of data type
        public int8_t dsize

        # start index for this type
        public int32_t start_index

        # attribute name
        public str name

        # number of slot
        public int32_t slot_num

    def __cinit__(self, str name, int8_t dtype, int8_t atype, int32_t slot_num):
        self.name = name
        self.dtype = dtype
        self.atype = atype
        self.slot_num = slot_num

        self.dsize = dtype_size_map[dtype]

# fused functions to access data
cdef graph_data_type get_value_from_ptr(graph_data_type *data, int32_t start_index, int32_t slot_index):
    return data[start_index + slot_index]

cdef void set_value_from_ptr(graph_data_type *data, int32_t start_index, int32_t slot_index, graph_data_type value):
    data[start_index + slot_index] = value


# functions to cast and access data
# TODO: good ways to refine this?
cdef object get_attr_value_from_array(view.array arr, int8_t dtype, int32_t aindex, int32_t sindex):
    if dtype == DT_BYTE:
        return get_value_from_ptr[int8_t](<int8_t *>arr.data, aindex, sindex)
    elif dtype == DT_SHORT:
        return get_value_from_ptr[int16_t](<int16_t *>arr.data, aindex, sindex)
    elif dtype == DT_INT32:
        return get_value_from_ptr[int32_t](<int32_t *>arr.data, aindex, sindex)
    elif dtype == DT_INT64:
        return get_value_from_ptr[int64_t](<int64_t *>arr.data, aindex, sindex)
    elif dtype == DT_FLOAT:
        return get_value_from_ptr[float](<float *>arr.data, aindex, sindex)

    return None

cdef void set_attr_value_from_array(view.array arr, int8_t dtype, int32_t aindex, int32_t sindex, object value):
    if dtype == DT_BYTE:
        set_value_from_ptr[int8_t](<int8_t *>arr.data, aindex, sindex, value)
    elif dtype == DT_SHORT:
        set_value_from_ptr[int16_t](<int16_t *>arr.data, aindex, sindex, value)
    elif dtype == DT_INT32:
        set_value_from_ptr[int32_t](<int32_t *>arr.data, aindex, sindex, value)
    elif dtype == DT_INT64:
        set_value_from_ptr[int64_t](<int64_t *>arr.data, aindex, sindex, value)
    elif dtype == DT_FLOAT:
        set_value_from_ptr[float](<float *>arr.data, aindex, sindex, value)

cdef class Graph:
    '''Graph used to hold attributes for both static and dynamic nodes.
    To initialize a graph, attributes must to be registered before setup.
    Example:
        Create a simple graph that with 10 static and 10 dynamic nodes, and attributes like "attr1", "attr2":
            static_node_num = 10
            dynamic_node_num = 10

            # init the graph object first
            graph = Graph(static_node_num, dynamic_node_num)

            # then register attributes
            # register an attribute named "attr1", its data type is float, can hold 1 value (slot)
            graph.register_attribute(GraphAttributeType.DYNAMIC, "attr1", GraphDataType.FLOAT, 1)
            
            # register an attribute named "attr2", its data type is int, can hold 2 value (slots)
            graph.register_attribute(GraphAttributeType.STATIC, "attr2", GraphDataType.INT32, 2)
            
            # then we can setup the graph for using
            graph.setup()
            
            # the graph is ready to accessing now
            # get an attribute (first slot) of a static node that id is 0
            a1 = graph.get_attribute(GraphAttributeType.DYNAMIC, 0, "attr1", 0)
            
            # set an attribute (2nd slot) of a dynamic node that id is 0
            graph.set_attribute(GraphAttributeType.DYNAMIC, 0, "attr1", 1, 123)
    Args:
        static_node_num (int): number of static nodes in graph
        dynamic_node_num (int): number of dynamic nodes in graph
    '''
    cdef:
        # actual data array
        view.array arr

        int32_t static_node_num
        int32_t dynamic_node_num

        # static nodes first, dynamic, then data
        int32_t dynamic_start_idx

        # attributes for partitions, key is: (partition_type, name)
        dict attr_map

        # total size in byte
        int64_t size

        # if graph already initialized
        bool is_initialized

    
    def __cinit__(self, int32_t static_node_num, int32_t dynamic_node_num):
        self.static_node_num = static_node_num
        self.dynamic_node_num = dynamic_node_num
        self.size = 0
        self.is_initialized = False

        self.attr_map = {}

    @property
    def memory_size(self)->int:
        """int: number of memory that cost (int byte)
        """
        return self.size

    @property
    def static_node_number(self) -> int:
        '''int: Number of static nodes in current graph'''
        return self.static_node_num

    @property
    def dynamic_node_number(self) -> int:
        '''int: Number of dynamic nodes in current graph'''
        return self.dynamic_node_num

    cpdef void register_attribute(self, int8_t atype, str name, int8_t data_type, int32_t slot_num):
        '''Register an attribute for nodes in graph, then can access the new attribute with get/set_attribute methods.
        NOTE: this method should be called before setup method
        Args:
            atype (GraphAttributeType): type of this attribute belongs to
            name (str): name of the attribute 
            data_type (GraphDataType): data type of attribute
            slot_num (int): how many slots of this attributes can hold
        Raises:
            GraphAttributeExistError: if the name already being registered
        '''
        if self.is_initialized == True:
            # cannot register after setup
            return

        # TODO: refactor later
        attr_key = (name, atype)

        if attr_key in self.attr_map:
            raise GraphAttributeExistError(f"Attribute name {name} already registered.")

        self.attr_map[attr_key] = GraphAttribute(name, data_type, atype, slot_num)

    cpdef void setup(self):
        '''Setup the graph with registered attributes
        Raises:
            GraphAttributeNotRegisteredError: if not registered any attribute
        '''
        if self.is_initialized:
            return

        if len(self.attr_map) == 0:
            raise GraphAttributeNotRegisteredError()

        # initial the fields
        cdef int32_t i = 0
        self.size = self.cal_graph_size()

        # allocate memory
        self.arr = view.array(shape=(1, self.size), itemsize=sizeof(int8_t), format="b")

        self.is_initialized = True

        for i in range(self.size):
            self.arr[0, i] = 0

    # TODO: refactor the node_id, to make it can be None value
    cpdef object get_attribute(self, int8_t atype, int32_t node_id, str attr_name, int32_t slot_index):
        '''Get specified attribute value with general way
        Args:
            atype (GraphAttributeType): type of attribute belongs to
            node_id (int): id the the resource node
            attr_name (str): name of accessing attribute
            slot_index (int): index of the attribute slot
        Returns:
            value of specified attribute slot
        Raises:
            GraphAttributeNotFoundError: specified attribute is not registered
        '''
        attr_key = (attr_name, atype)

        if attr_key not in self.attr_map:
            raise GraphAttributeNotFoundError(f"attribute {attr_name} is not registered.")

        cdef GraphAttribute attr = self.attr_map[attr_key]

        if atype == AT_GENERAL or node_id is None:
            node_id = 0
        
        # index of current slot
        cdef int32_t aindex = attr.start_index + (attr.slot_num * node_id)
        
        return get_attr_value_from_array(self.arr, attr.dtype, aindex, slot_index)

    cpdef set_attribute(self, int8_t atype, int32_t node_id, str attr_name, int32_t slot_index, object value):
        '''Set specified attribute value
        Args:
            atype (GraphAttributeType): type of attribute belongs to
            node_id (int): id the the resource node
            attr_name (str): name of accessing attribute
            slot_index (int): index of the attribute slot        
            value (float/int): value to set
        Raises:
            GraphAttributeNotFoundError: specified attribute is not registered
        '''
        attr_key = (attr_name, atype)

        if attr_key not in self.attr_map:
            raise GraphAttributeNotFoundError(f"attribute {attr_name} is not registered.")

        cdef GraphAttribute attr = self.attr_map[attr_key]
        
        if atype == AT_GENERAL or node_id is None:
            node_id = 0

        cdef int32_t aindex = attr.start_index + (attr.slot_num * node_id)

        set_attr_value_from_array(self.arr, attr.dtype, aindex, slot_index, value)

    cpdef void reset(self):
        '''Reset all the attributes to default value'''
        cdef int64_t i = 0

        for i in range(self.size):
            self.arr[0, i] = 0

    cpdef void save(self, bytes path):
        """Dump current graph data into bytes.
        The bytes contains following part:
        2. version: 1 bytes
        3. endian type: 1 byte (0 or 1)
        4. size (in byte): 8 bytes
        5. data
        """
        cdef int8_t version = 1 
        cdef int8_t etype
        cdef int8_t *data_ptr = <int8_t*>self.arr.data
        cdef FILE *fp

        if sys.byteorder == "little":
            etype = 0 # 0 means little endian, 1 means big endian
        else:
            etype = 1

        fp = fopen(path, "w")

        if not fp:
            # TODO: exception later
            return
        
        fwrite(&version, sizeof(int8_t), 1, fp)
        fwrite(&etype, sizeof(int8_t), 1, fp)
        fwrite(data_ptr, sizeof(int8_t), self.size, fp)

        fclose(fp)    
    
    cpdef load(self, bytes path):
        # read name, version, type, size

        # read all the data into <int8_t*>self.arr.data

        # go through all the attributes and swap bytes if endian is not same with current

        cdef FILE *fp
        cdef int8_t[2] buffer
        cdef int8_t *data_ptr = <int8_t*>self.arr.data

        fp = fopen(path, "r")

        # read header
        fread(buffer, sizeof(int8_t), 2, fp)

        # buffer[0] version
        # buffer[1] endian type

        # TODO: big endian support later
        fread(data_ptr, sizeof(int8_t), self.size, fp)

        fclose(fp)

    cdef int32_t cal_graph_size(self):
        """
        Calculate size of partition, and update attribute start index
        """
        if len(self.attr_map) == 0:
            return 0

        cdef list attr_list = None  
        cdef int32_t i = 0
        cdef int32_t size = 0
        cdef GraphAttribute attr = None

        attr_list = [attr for _, attr in self.attr_map.items()]
        attr_list.sort(key=lambda x: x.dtype)

        cdef int32_t cur_size = attr_list[0].dsize
        cdef int32_t pre_size = cur_size
        cdef int32_t attr_total_size = 0
        cdef int32_t node_type_factor = 1

        for i in range(len(attr_list)):
            attr = attr_list[i]

            if attr.atype == AT_STATIC:
                node_type_factor = self.static_node_num
            elif attr.atype == AT_DYNAMIC:
                node_type_factor = self.dynamic_node_num
            else:
                node_type_factor = 1

            attr_total_size = attr.dsize * attr.slot_num * node_type_factor

            cur_size = attr.dsize

            if cur_size != pre_size:
                # keep increasing the size if the data type size not changed
                # or checking if we need padding
                size = ceil(size/cur_size) * cur_size
        
            pre_size = cur_size
            attr.start_index = int(size / cur_size)
            size += attr_total_size

        return size


cdef class SnapshotList:
    '''SnapshotList used to hold list of snapshots that taken from Graph object at a certain tick.

    SnapshotList only provide interface to get data, cannot set data.

    Examples:
        it is recommended to use slice to query attributes.
        . snapshot_list.static_nodes[[tick list]: [node id list]: ([attribute names], [attribute slot list])]
        . snapshot_list.dynamic_nodes[[tick list]: [node id list]: ([attribute names], [attribute slot list])]

        all the list parameter can be a single value.

        if tick or node id list is None, then means query all.

        # query 1st slot value of attribute "a1" for node 1 at all the ticks
        snapshot_list.static_nodes[: 1: ("a1", 0)]

        # query 1st and 2nd slot value of attribute "a1" for all the nodes at tick 0
        snapshot_list.static_nodes[0: : ("a1", [0, 1])]

        # query 1st slot value for attribute "a1" and "a2" for all the nodes at all the ticks
        snapshot_list.static_nodes[:: (["a1", "a2"], 0)]

        # query a matrix at tick 0
        snapshot_list.matrix[0: "m1"]

        # query matrix at tick 0 and 1
        snapshot_list.matrix[[0, 1]: "m1"]
    '''
    cdef:
        Graph graph

        # number of snapshot
        int32_t size

        # memory size of graph
        int64_t graph_size

        # actual data 
        view.array arr

        # memory view of array
        int8_t[:, :, :] data 

        # index and tick for snapshot query
        int32_t start_index
        int32_t end_index
        int32_t start_tick
        int32_t end_tick

        # internal tick 
        int32_t tick

        SnapshotNodeAccessor static_acc
        SnapshotNodeAccessor dynamic_acc
        SnapshotGeneralAccessor general_acc


    def __cinit__(self, Graph graph, int32_t size):
        self.graph = graph
        self.size = size
        self.graph_size = graph.size
        self.start_index = -1
        self.end_index = -1
        self.start_tick = 0
        self.end_tick = -1
        
        self.arr = view.array(shape=(size, 1, self.graph_size), itemsize=sizeof(int8_t), format="b")
        self.data = self.arr

        self.static_acc = SnapshotNodeAccessor(AT_STATIC, self.graph.static_node_num, self)
        self.dynamic_acc = SnapshotNodeAccessor(AT_DYNAMIC, self.graph.dynamic_node_num, self)
        self.general_acc = SnapshotGeneralAccessor(self)

    @property
    def memory_size(self)->int:
        """int: number of memory that cost (int byte)
        """
        return self.size * self.graph_size
        
    @property
    def static_nodes(self) -> SnapshotNodeAccessor:
        '''Same as dynamic_nodes, but for static nodes'''
        return self.static_acc

    @property
    def dynamic_nodes(self) -> SnapshotNodeAccessor:
        '''Slice interface to query attribute value of dynamic nodes.

        The slice like [tick: id: (attribute name, slot index]

        tick: tick to query, can be a list
        id: id of dynamic nodes to query, can be a list
        attribute name: registered attribute to query, can be a list
        slot index: slot to query, can be a list

        Examples:
            # query 1st slot value of attribute "a1" for node 1 at all the ticks
            snapshot_list.dynamic_nodes[: 1: ("a1", 0)]

            # query 1st and 2nd slot value of attribute "a1" for all the nodes at tick 0
            snapshot_list.dynamic_nodes[0: : ("a1", [0, 1])]

            # query 1st slot value for attribute "a1" and "a2" for all the nodes at all the ticks
            snapshot_list.dynamic_nodes[:: (["a1", "a2"], 0)]

        Returns:
            np.ndarray: states numpy array (1d)
        '''
        return self.dynamic_acc

    @property
    def general(self) -> SnapshotGeneralAccessor:
        '''Slice interface to access general attributes

        The slice like [tick: name]

        tick: tick to query, can be a list
        name: name of the matrix to query

        Examples:
            # query a matrix at tick 0
            snapshot_list.matrix[0: "m1"]

            # query matrix at tick 0 and 1
            snapshot_list.matrix[[0, 1]: "m1"]
        '''
        return self.general_acc

    @property
    def matrix(self):
        '''Same with general, used to fit previouse interface
        '''
        return self.general_acc

    @property
    def dynamic_node_number(self) -> int:
        '''int: Dynamic node number in each snapshot'''
        return self.graph.dynamic_node_num

    @property
    def static_node_number(self) -> int:
        '''int: Static node number in each snapshot'''
        return self.graph.static_node_num

    @property
    def ticks(self):
        return [i for i in range(self.start_tick, self.end_tick+1)]

    @property
    def attributes(self):
        '''List of the attributes information in current snapshot
        Returns:
            list: A list of attribute details
        '''
        result = []

        for attr_key, attr in self.graph.attr_map.items():
            result.append({
                "name": attr.name,
                "slot length": attr.slot_num,
                "attribute type": attr.atype
            })

        return result        

    cpdef reset(self):
        """Reset snapshot list
        """
        self.start_index = -1
        self.end_index = -1
        self.start_tick = 0
        self.end_tick = -1
        
    cpdef void insert_snapshot(self, int32_t tick):
        '''Insert a snapshot from graph'''
        # check if the tick is valid
        if tick < self.end_tick or tick > self.end_tick + 1:
            raise SnapshotInvalidTick(f"Invalid tick {tick} to take snapshot, we do not support override previous data")

        cdef int8_t[:, :] t= self.graph.arr

        # if tick is same with current end_tick, then means we need to override
        # NOTE: we do not support override previous snapshot now
        if tick == self.end_tick:
            self.data[tick::] = t
            
            return

        self.end_index += 1

        # back to the beginning if we reach the end
        if self.end_index >= self.size:
            self.end_index = 0

        # move start index if read the beginning
        if self.end_index == self.start_index:
            self.start_index += 1
            self.start_tick += 1

        # correct start index
        if self.start_index >= self.size or self.start_index == -1:
            self.start_index = 0

        self.data[self.end_index::] = t

        self.end_tick = tick

    cpdef np.ndarray get_node_attributes(self, int8_t atype, list ticks, list ids, list attr_names, list attr_indices, float default_value):
        '''Query states from snapshot list.
        Note:
            It is recommended that use slice interface instead of this raw method.
        
        Examples:
            Query value at 1st slot of attributes "a1" and "a2" at tick (0, 1, 2) for dynamic nodes (3, 4, 5)
            ...
            ticks = [0, 1, 2]
            node_ids = [3, 4, 5]
            attrs = ["a1", "a2"]
            slots = [0, ]
            # this will return  3*3*2*1 size of numpy array (dim=1)
            state = snapshotlist.get_attributes(ResourceNodeType.DYNAMIC, ticks, node_ids, attrs, slots)
            # if you are not sure about the slot length of an attribute
            slots = snapshotlist.get_slot_length("a1")
        Args:
            atype (GraphAttributeType): type of resource node, static or dynamic
            ticks (list[int]): list of tick to query, if the tick not available, then related value will be 0
            node_ids (list[int]): list of node id, if the id not exist, then the related value will be 0
            attr_names (list[str]): attribute names to query, if the attribute not exist, then the related value will be 0
            attr_indices (list[int]): slots of attributes to query, if the index is large than registered size, then related value will be set 0 for it
            default_value(float): default value if quering is invalid
        Returns:
            np.ndarray: numpy array (dim=1, size=len(ticks) * len(node_ids) * len(attribute_names) * len(attribute_indices)) with result
        '''
        cdef int32_t ticks_length = len(ticks)
        cdef int32_t ids_length    = len(ids)
        cdef int32_t attr_length = len(attr_names)
        cdef int32_t index_length = len(attr_indices)

        cdef int32_t tick
        cdef int32_t node_id
        cdef str attr_name
        cdef int32_t attr_index
        cdef GraphAttribute attr
        cdef int8_t attr_dtype
        cdef int32_t ridx = 0 # index of result
        cdef int32_t tindex = 0 # index of tick
        cdef int32_t aindex = 0 # index of attribute
        cdef max_node_num = self.graph.static_node_num
        cdef np.ndarray result = np.zeros(ticks_length * ids_length * attr_length * index_length, dtype=np.float32)
        cdef float[:] result_view = result
        
        attr_key = None

        if atype == AT_DYNAMIC:
            max_node_num = self.graph.dynamic_node_num
        elif atype == AT_GENERAL:
            max_node_num = 1

        for tick in ticks:
            if not (self.start_tick <= tick <= self.end_tick):
                ridx += ids_length * attr_length * index_length

                continue

            tindex = self.start_index + (tick - self.start_tick)

            if tindex >= self.size:
                tindex = tindex - self.size

            for node_id in ids:
                if node_id < 0 or node_id >= max_node_num:
                    ridx += attr_length * index_length

                    continue

                for attr_name in attr_names:
                    for attr_index in attr_indices:
                        attr_key = (attr_name, atype)

                        if attr_key in self.graph.attr_map:
                            attr = self.graph.attr_map[attr_key]
                            attr_dtype = attr.dtype

                            if attr_index < attr.slot_num:
                                
                                aindex = <int32_t>(tindex * self.graph.size / attr.dsize) + attr.start_index + (attr.slot_num * node_id)
 
                                v = get_attr_value_from_array(self.arr, attr_dtype, aindex, attr_index)

                                result_view[ridx] = <float>v

                        ridx += 1

        return result

    def get_general_attribute(self, list ticks, str attr_name, float default_value=0):
        '''Get value of a attribute for specified ticks.
        NOTE: it is recommended to use slice method (snapshot_list.matrix) to query.
        Args:
            ticks (list): list of ticks to query
            attr_name (str): name of attribute name to query
            default_value(float): default value if quering is invalid
        '''
        attr_key = (attr_name, AT_GENERAL)

        if attr_key not in self.graph.attr_map:
            return None

        cdef GraphAttribute attr = self.graph.attr_map[attr_key]
        cdef int32_t length = attr.slot_num
        cdef int32_t ticks_length = len(ticks)

        cdef np.ndarray result = np.zeros((ticks_length, length), dtype=np.float32)
        cdef float[:, :] ret_view = result
        cdef int8_t attr_dtype = attr.dtype

        cdef int32_t i, j
        cdef int32_t tick
        cdef int32_t tindex
        cdef int32_t aindex


        for i in range(ticks_length):
            tick = ticks[i]

            if not (self.start_tick <= tick <= self.end_tick):
                continue

            tindex = self.start_index + (tick - self.start_tick)

            if tindex >= self.size:
                tindex = tindex - self.size

            aindex = <int32_t>(tindex * self.graph.size / attr.dsize) + attr.start_index

            for j in range(length):
                v = get_attr_value_from_array(self.arr, attr_dtype, aindex, j)
                
                ret_view[i, j] = <float>v

        return result



    def __len__(self):
        return self.end_tick - self.start_tick + 1
    
    
cdef class SnapshotNodeAccessor:
    """
    Wrapper to access node attributes with slice interface
    """
    cdef:
        int8_t atype
        int32_t node_num
        SnapshotList ss

    def __cinit__(self, int8_t atype, int32_t node_num, SnapshotList ss):
        self.atype = atype
        self.node_num = node_num
        self.ss = ss

    def __len__(self):
        return len(self.ss)

    def __setitem__(self, item, value):
        pass

    def __getitem__(self, slice item):
        '''Query states from snapshot'''
        cdef list ticks
        cdef list id_list
        cdef list attr_names
        cdef list attr_ndices
        cdef int32_t i

        if item.start is None:
            ticks = self.ss.ticks
        else:
            if type(item.start) is not list:
                ticks = [item.start]
            else:
                ticks = item.start

        if item.stop is None:
            id_list = [i for i in range(self.node_num)]
        else:
            if type(item.stop) is not list:
                id_list = [item.stop]
            else:
                id_list = item.stop

        if item.step is None or len(item.step) < 2:
            return None # TODO: exception later

        cdef tuple attributes = item.step

        # correct parameters
        if type(attributes[0]) is not list:
            attr_names = [attributes[0]]
        else:
            attr_names = attributes[0]

        if type(attributes[1]) is not list:
            attr_indices = [attributes[1]]
        else:
            attr_indices = attributes[1]
        
        return self.ss.get_node_attributes(self.atype, ticks, id_list, attr_names, attr_indices, 0)

cdef class SnapshotGeneralAccessor:
    """
    Wrapper to access general data with slice interface
    """
    cdef:
        SnapshotList ss

    def __cinit__(self, SnapshotList ss):
        self.ss = ss

    def __getitem__(self, slice item):
        cdef list ticks
        cdef str attr_name = item.stop

        if type(item.start) is not list:
            ticks = [item.start]
        else:
            ticks = item.start

        return self.ss.get_general_attribute(ticks, attr_name, 0)
