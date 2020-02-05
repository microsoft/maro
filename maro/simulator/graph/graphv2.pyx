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

        public int8_t dsize

        # start index for this type
        public int64_t start_index

        public str name

        public int32_t slot_num

    def __cinit__(self, str name, int8_t dtype, int32_t slot_num):
        self.name = name
        self.dtype = dtype
        self.slot_num = slot_num

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
        view.array arr

        int16_t static_node_num
        int16_t dynamic_node_num

        # static nodes first, dynamic, then data
        int32_t dynamic_start_idx

        # attributes for partitions
        dict attr_map

        # total size in byte
        int64_t size

        #
        bool is_initialized

    
    def __cinit__(self, int16_t static_node_num, int16_t dynamic_node_num):
        self.static_node_num = static_node_num
        self.dynamic_node_num = dynamic_node_num
        self.size = 0
        self.is_initialized = False

        self.attr_map = {}

    cpdef void reg_attr(self, int8_t partition_type, str name, int8_t data_type, int32_t slot_num):
        if self.is_initialized == True:
            return

        attr_key = (name, partition_type)

        if attr_key in self.attr_map:
            return

        self.attr_map[attr_key] = GraphAttribute(name, data_type, slot_num)

    cpdef void setup(self):
        cdef int32_t total_size = self.cal_partition_size()

        print("total size", total_size)

        self.arr = view.array(shape=(1, total_size), itemsize=sizeof(int8_t), format="b")

        self.is_initialized = True

        for _, attr in self.attr_map.items():
            print(attr.dsize, attr.start_index, attr.dtype)

        cdef int32_t i = 0

        for i in range(total_size):
            self.arr[0, i] = 0

    cdef int32_t cal_partition_size(self):
        """
        Calculate size of partition
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

            if attr.dtype == PartitionType.STATIC_NODE:
                node_type_factor = self.static_node_num
            elif attr.dtype == PartitionType.DYNAMIC_NODE:
                node_type_factor = self.dynamic_node_num

            attr_total_size = attr.dsize * attr.slot_num * node_type_factor

            cur_size = attr.dsize

            if cur_size != pre_size:
                # keep increasing the size if the data type size not changed
                # or checking if we need padding
                size = ceil(size/cur_size) * cur_size
        
            pre_size = cur_size
            attr.start_index = int(size / cur_size)
            size += attr_total_size

            print(attr.name, attr.dsize, attr.slot_num, attr.start_index)


        return size

    cpdef get_attr(self, int8_t node_type, int16_t node_id, str attr_name, int32_t slot_index):
        attr_key = (attr_name, node_type)

        attr = self.attr_map[attr_key]
        
        if attr.dtype == AttributeType.BYTE:
            return get_graph_attr_value[int8_t](<int8_t *>self.arr.data, attr.start_index + (attr.slot_num * node_id), slot_index)
        elif attr.dtype == AttributeType.SHORT:
            return get_graph_attr_value[int16_t](<int16_t *>self.arr.data, attr.start_index + (attr.slot_num * node_id), slot_index)
        elif attr.dtype == AttributeType.INT32:
            return get_graph_attr_value[int32_t](<int32_t *>self.arr.data, attr.start_index + (attr.slot_num * node_id), slot_index)
        elif attr.dtype == AttributeType.INT64:
            return get_graph_attr_value[int64_t](<int64_t *>self.arr.data, attr.start_index + (attr.slot_num * node_id), slot_index)
        elif attr.dtype == AttributeType.FLOAT32:
            return get_graph_attr_value[float](<float *>self.arr.data, attr.start_index + (attr.slot_num * node_id), slot_index)
        elif attr.dtype == AttributeType.DOUBLE:
            return get_graph_attr_value[double](<double *>self.arr.data, attr.start_index + (attr.slot_num * node_id), slot_index)

    cpdef set_attr(self, int8_t node_type, int16_t node_id, str attr_name, int8_t slot_index, object value):
        attr_key = (attr_name, node_type)

        attr = self.attr_map[attr_key]

        if attr.dtype == AttributeType.BYTE:
            set_graph_attr_value[int8_t](<int8_t *>self.arr.data, attr.start_index + (attr.slot_num * node_id), slot_index, value)
        elif attr.dtype == AttributeType.SHORT:
            set_graph_attr_value[int16_t](<int16_t *>self.arr.data, attr.start_index + (attr.slot_num * node_id), slot_index, value)
        elif attr.dtype == AttributeType.INT32:
            set_graph_attr_value[int32_t](<int32_t *>self.arr.data, attr.start_index + (attr.slot_num * node_id), slot_index, value)
        elif attr.dtype == AttributeType.INT64:
            set_graph_attr_value[int64_t](<int64_t *>self.arr.data, attr.start_index + (attr.slot_num * node_id), slot_index, value)
        elif attr.dtype == AttributeType.FLOAT32:
            set_graph_attr_value[float](<float *>self.arr.data, attr.start_index + (attr.slot_num * node_id), slot_index, value)
        elif attr.dtype == AttributeType.DOUBLE:
            set_graph_attr_value[double](<double *>self.arr.data, attr.start_index + (attr.slot_num * node_id), slot_index, value)

def test():
    cdef int8_t a = 1
    cdef int8_t b = 12

    print(sizeof(int8_t))
    print(sizeof(int16_t))
    print(sizeof(int32_t))

    return a + b

def test_byte_cast():
    cdef view.array arr = view.array(shape=(1, 100), itemsize=sizeof(int8_t), format="c")

    cdef int8_t[:, :] v = arr

    v[0, 0] = 1

    cdef char *aptr = <char *>arr.data

    cdef float *bptr = <float *>arr.data

    bptr[1] = 3.0

    cdef int16_t *cptr = <int16_t *>arr.data

    cptr[1] = 12


    return v[0, 0], v[0, 1], v[0, 2], v[0, 3], v[0, 4], v[0, 5], v[0, 6], v[0, 7], bptr[1]


