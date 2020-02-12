#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython

from enum import IntEnum
from cython cimport view
from cpython cimport bool
from math import ceil

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t

ctypedef fused graph_data_type:
    int8_t
    int16_t
    int32_t
    int64_t
    float
    double

class AttributeType(IntEnum):
    BYTE = 0
    SHORT = 1
    INT32 = 2
    INT64 = 3
    FLOAT32 = 4
    DOUBLE = 5
    
class PartitionType(IntEnum):
    STATIC_NODE = 0
    DYNAMIC_NODE = 1
    GENERAL = 2

cdef class GraphAttribute:
    cdef:
        # data type
        public int8_t dtype

        # partition type
        public int8_t ptype

        # size of data type
        public int8_t dsize

        # start index for this type
        public int64_t start_index

        # attribute name
        public str name

        # number of slot
        public int32_t slot_num

    def __cinit__(self, str name, int8_t dtype, int8_t ptype, int32_t slot_num):
        self.name = name
        self.dtype = dtype
        self.ptype = ptype
        self.slot_num = slot_num

        # TODO: refactor later
        if dtype == AttributeType.BYTE:
            self.dsize = 1
        elif dtype == AttributeType.SHORT:
            self.dsize = 2
        elif dtype == AttributeType.INT32:
            self.dsize = 4
        elif dtype == AttributeType.INT64:
            self.dsize = 8
        elif dtype == AttributeType.FLOAT32:
            self.dsize = 4
        elif dtype == AttributeType.DOUBLE:
            self.dsize = 8

# fused functions to access data
cdef graph_data_type get_graph_attr_value(graph_data_type *data, int32_t start_index, int32_t slot_index):
    return data[start_index + slot_index]

cdef void set_graph_attr_value(graph_data_type *data, int32_t start_index, int32_t slot_index, graph_data_type value):
    data[start_index + slot_index] = value

cdef class Graph:
    """
    Each Graph object will contain 3 partitions for different data:
    1. static node attributes
    2. dynamic node attributes
    3. general data partition: contains matrix(s) or other data that not related to any node type
    """
    cdef:
        # actual data array
        view.array arr

        int16_t static_node_num
        int16_t dynamic_node_num

        # static nodes first, dynamic, then data
        int32_t dynamic_start_idx

        # attributes for partitions, key is: (partition_type, name)
        dict attr_map

        # total size in byte
        int64_t size

        # if graph already initialized
        bool is_initialized

    
    def __cinit__(self, int16_t static_node_num, int16_t dynamic_node_num):
        self.static_node_num = static_node_num
        self.dynamic_node_num = dynamic_node_num
        self.size = 0
        self.is_initialized = False

        self.attr_map = {}

    cpdef void reg_attr(self, int8_t partition_type, str name, int8_t data_type, int32_t slot_num):
        """
        Register attribute
        """
        if self.is_initialized == True:
            return

        # TODO: refactor later
        attr_key = (name, partition_type)

        if attr_key in self.attr_map:
            return

        self.attr_map[attr_key] = GraphAttribute(name, data_type, partition_type, slot_num)

    cpdef void setup(self):
        """
        Setup the graph memory based on registered attributes
        """
        cdef int32_t total_size = self.cal_partition_size()
        self.size = total_size

        # allocate memory
        self.arr = view.array(shape=(1, total_size), itemsize=sizeof(int8_t), format="b")

        self.is_initialized = True

        # initial the fields
        cdef int32_t i = 0

        for i in range(total_size):
            self.arr[0, i] = 0

    cdef int32_t cal_partition_size(self):
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
        attr_list.sort(key=lambda x: x.dsize)

        cdef int32_t cur_size = attr_list[0].dsize
        cdef int32_t pre_size = cur_size
        cdef int32_t attr_total_size = 0
        cdef int32_t node_type_factor = 1

        for i in range(len(attr_list)):
            attr = attr_list[i]

            if attr.ptype == PartitionType.STATIC_NODE:
                node_type_factor = self.static_node_num
            elif attr.ptype == PartitionType.DYNAMIC_NODE:
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

    # TODO: refactor the node_id, to make it can be None value
    cpdef get_attr(self, int8_t node_type, int16_t node_id, str attr_name, int32_t slot_index):
        """
        Get value of attribute
        """
        attr_key = (attr_name, node_type)

        attr = self.attr_map[attr_key]

        if node_type == PartitionType.GENERAL or node_id is None:
            node_id = 0
        
        aindex = attr.start_index + (attr.slot_num * node_id)
        
        # TODO: try to refactor this chunck
        if attr.dtype == AttributeType.BYTE:
            return get_graph_attr_value[int8_t](<int8_t *>self.arr.data, aindex, slot_index)
        elif attr.dtype == AttributeType.SHORT:
            return get_graph_attr_value[int16_t](<int16_t *>self.arr.data, aindex, slot_index)
        elif attr.dtype == AttributeType.INT32:
            return get_graph_attr_value[int32_t](<int32_t *>self.arr.data, aindex, slot_index)
        elif attr.dtype == AttributeType.INT64:
            return get_graph_attr_value[int64_t](<int64_t *>self.arr.data, aindex, slot_index)
        elif attr.dtype == AttributeType.FLOAT32:
            return get_graph_attr_value[float](<float *>self.arr.data, aindex, slot_index)
        elif attr.dtype == AttributeType.DOUBLE:
            return get_graph_attr_value[double](<double *>self.arr.data, aindex, slot_index)

    cpdef set_attr(self, int8_t node_type, int16_t node_id, str attr_name, int8_t slot_index, object value):
        attr_key = (attr_name, node_type)

        attr = self.attr_map[attr_key]
        
        if node_type == PartitionType.GENERAL or node_id is None:
            node_id = 0

        aindex = attr.start_index + (attr.slot_num * node_id)

        if attr.dtype == AttributeType.BYTE:
            set_graph_attr_value[int8_t](<int8_t *>self.arr.data, aindex, slot_index, value)
        elif attr.dtype == AttributeType.SHORT:
            set_graph_attr_value[int16_t](<int16_t *>self.arr.data, aindex, slot_index, value)
        elif attr.dtype == AttributeType.INT32:
            set_graph_attr_value[int32_t](<int32_t *>self.arr.data, aindex, slot_index, value)
        elif attr.dtype == AttributeType.INT64:
            set_graph_attr_value[int64_t](<int64_t *>self.arr.data, aindex, slot_index, value)
        elif attr.dtype == AttributeType.FLOAT32:
            set_graph_attr_value[float](<float *>self.arr.data, aindex, slot_index, value)
        elif attr.dtype == AttributeType.DOUBLE:
            set_graph_attr_value[double](<double *>self.arr.data, aindex, slot_index, value)


    cpdef reset(self):
        cdef int64_t i = 0

        for i in range(self.size):
            self.arr[0, i] = 0

    @property
    def static_node_num(self) -> int:
        return self.static_node_num

    @property
    def dynamic_node_num(self) -> int:
        return self.dynamic_node_num


cdef class SnapshotList:
    cdef:
        Graph graph

        # total size of memory
        int32_t size

        # memory size of graph
        int64_t graph_size

        # actual data 
        view.array arr

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


    def __cinit__(self, int32_t size, Graph graph):
        self.graph = graph
        self.size = size
        self.graph_size = graph.size
        self.start_index = -1
        self.end_index = -1
        self.start_tick = 0
        self.end_tick = -1
        self.tick = -1
        
        self.arr = view.array(shape=(size, 1, self.graph_size), itemsize=sizeof(int8_t), format="b")

        self.static_acc = SnapshotNodeAccessor(PartitionType.STATIC_NODE, self.graph.static_node_num, self)
        self.dynamic_acc = SnapshotNodeAccessor(PartitionType.DYNAMIC_NODE, self.graph.dynamic_node_num, self)
        self.general_acc = SnapshotGeneralAccessor(self)
        
    @property
    def static_nodes(self) -> SnapshotNodeAccessor:
        return self.static_acc

    @property
    def dynamic_nodes(self) -> SnapshotNodeAccessor:
        return self.dynamic_acc

    @property
    def general(self) -> SnapshotGeneralAccessor:
        """
        Access general data
        """
        return self.general_acc

    @property
    def maxtrix(self):
        """
        Access general data and return the result as matrix (1, n)
        """
        pass

    cpdef reset(self):
        self.start_index = -1
        self.end_index = -1
        self.start_tick = 0
        self.end_tick = -1
        self.tick = -1

    cpdef void insert_snapshot(self):
        cdef int8_t[:, :] t= self.graph.arr
        
        self.end_index += 1
        self.tick += 1

        # back to the beginning if we reach the end
        if self.end_index >= self.size:
            self.end_index = 0

        if self.end_index == self.start_index:
            self.start_index += 1
            self.start_tick += 1

        if self.start_index >= self.size or self.start_index == -1:
            self.start_index = 0

        self.data[self.end_index::] = t

        self.end_tick = self.tick

    cpdef np.ndarray get_node_attrs(self, int8_t node_type, list ticks, list ids, list attr_names, list attr_indices, float default_value):
        
        # used to check if id list is valid
        
        # check id
    
        # check ticks

        # check attributes
        cdef int32_t ticks_length = len(ticks)
        cdef int32_t ids_length    = len(ids)
        cdef int32_t attr_length = len(attr_names)
        cdef int32_t index_length = len(attr_indices)

        cdef int32_t tick
        cdef int32_t node_id
        cdef str attr_name
        cdef int32_t attr_index

        cdef np.ndarray result = np.zeros(ticks_length * ids_length * attr_length * index_length, dtype=np.float32)

        cdef float[:] result_view = result

        cdef int32_t ridx = 0
        cdef int32_t tindex = 0
        cdef max_node_num = self.graph.static_node_num
        attr_key = None

        if node_type == PartitionType.DYNAMIC_NODE:
            max_node_num = self.graph.dynamic_node_num
        elif node_type == PartitionType.GENERAL:
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
                        attr_key = (attr_name, node_type)

                        if attr_key in self.graph.attr_map:
                            attr = self.graph.attr_map[attr_key]
                            attr_dtype = attr.dtype

                            if attr_index < attr.slot_num:
                                
                                aindex = tindex * self.graph.size / attr.dsize + attr.start_index
                                
                                # TODO: refactor later
                                if attr_dtype == AttributeType.BYTE:
                                    v = get_graph_attr_value[int8_t](<int8_t*>self.arr.data, aindex, attr_index)
                                elif attr_dtype == AttributeType.SHORT:
                                    v = get_graph_attr_value[int16_t](<int16_t*>self.arr.data, aindex, attr_index)
                                elif attr_dtype == AttributeType.INT32:
                                    v = get_graph_attr_value[int32_t](<int32_t*>self.arr.data, aindex, attr_index)
                                elif attr_dtype == AttributeType.INT64:
                                    v = get_graph_attr_value[int64_t](<int64_t*>self.arr.data, aindex, attr_index)
                                elif attr_dtype == AttributeType.FLOAT32:
                                    v = get_graph_attr_value[float](<float*>self.arr.data, aindex, attr_index)
                                elif attr_dtype == AttributeType.DOUBLE:
                                    v = get_graph_attr_value[double](<double*>self.arr.data, aindex, attr_index)

                                result_view[ridx] = <float>v

                        ridx += 1

        return result

    def get_general_attr(self, list ticks, str attr_name, float default_value=0):
        """
        
        """
        attr_key = (attr_name, <int8_t>PartitionType.GENERAL)

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
                if attr_dtype == AttributeType.BYTE:
                    v = get_graph_attr_value[int8_t](<int8_t*>self.arr.data, aindex, j)
                elif attr_dtype == AttributeType.SHORT:
                    v = get_graph_attr_value[int16_t](<int16_t*>self.arr.data, aindex, j)
                elif attr_dtype == AttributeType.INT32:
                    v = get_graph_attr_value[int32_t](<int32_t*>self.arr.data, aindex, j)
                elif attr_dtype == AttributeType.INT64:
                    v = get_graph_attr_value[int64_t](<int64_t*>self.arr.data, aindex, j)
                elif attr_dtype == AttributeType.FLOAT32:
                    v = get_graph_attr_value[float](<float*>self.arr.data, aindex, j)
                elif attr_dtype == AttributeType.DOUBLE:
                    v = get_graph_attr_value[double](<double*>self.arr.data, aindex, j)
                
                ret_view[i, j] = <float>v

        return result

    
    def ticks(self):
        return [i for i in range(self.end_tick-self.cur_size+1, self.end_tick+1)]

    def __len__(self):
        return self.cur_size


    
    
cdef class SnapshotNodeAccessor:
    """
    Wrapper to access node attributes with slice interface
    """
    cdef:
        int8_t node_type
        int16_t node_num
        SnapshotList ss

    def __cinit__(self, int8_t node_type, int16_t node_num, SnapshotList ss):
        self.node_type = node_type
        self.node_num = node_num
        self.ss = ss

    def __len__(self):
        return len(self.ss)

    def __setitem__(self, item, value):
        pass

    def __getitem__(self, slice item):
        cdef list ticks
        cdef list id_list
        cdef list attribute_names
        cdef list attribute_indices
        cdef int32_t i

        if item.start is None:
            ticks = self.ss.ticks()
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
            attribute_names = [attributes[0]]
        else:
            attribute_names = attributes[0]

        if type(attributes[1]) is not list:
            attribute_indices = [attributes[1]]
        else:
            attribute_indices = attributes[1]
        
        return self.ss.get_node_attrs(self.node_type, ticks, id_list, attribute_names, attribute_indices, 0)

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

        return self.ss.get_general_attr(ticks, attr_name, 0)
