#cython: language_level=3

from enum import IntEnum
from cpython cimport array, bool
import numpy as np
cimport cython
cimport numpy as np

ctypedef unsigned char BYTE
ctypedef float         FLOAT
ctypedef int           INT
ctypedef unsigned int  UINT
ctypedef int           BOOL
ctypedef int           ERROR
ctypedef UINT          NODE_TYPE

INT_DTYPE = np.intc
FLOAT_DTYPE = np.float32

# we use following const value to speedup index calculation
cdef NODE_TYPE_STATIC = 0
cdef NODE_TYPE_DYNAMIC = 1

cdef ATTR_TYPE_FLOAT = 0
cdef ATTR_TYPE_INT = 1
# int matrix
# TODO: any way to make it simple?
cdef ATTR_TYPE_INT_MAT = 2


class FrameAttributeType(IntEnum):
    '''Data type of registered attribute, can be FLOAT or INT now'''
    FLOAT = ATTR_TYPE_FLOAT
    INT = ATTR_TYPE_INT
    INT_MAT = ATTR_TYPE_INT_MAT


class FrameNodeType(IntEnum):
    '''Type of nodes in frame, STATIC and DYNAMIC for now'''
    STATIC = NODE_TYPE_STATIC
    DYNAMIC = NODE_TYPE_DYNAMIC


cdef class FrameAttribute:
    '''Used to wrapper attribute accessing information internally'''
    cdef:
        public UINT attribute_type
        public UINT size
        public UINT start_index
        public str name

        # for matrix attribute
        # TODO: refine later
        public UINT row_num
        public UINT column_num

    def __cinit__(self, str name, UINT attribute_type, UINT size, UINT start_index,  UINT row_num, UINT column_num):
        self.attribute_type = attribute_type
        self.name = name
        self.size = size
        self.start_index = start_index
        self.row_num = row_num
        self.column_num = column_num


class FrameError(Exception):
    '''Base exception of Frame'''
    def __init__(self, msg):
        self.message = msg


class FrameMemoryError(FrameError):
    '''Exception when we meet an memory issue when accessing Frame'''
    def __init__(self, msg):
        super().__init__(msg)


class FrameInvalidAccessError(FrameError):
    '''Exception that for invalid accessing, such as wrong index, etc.'''
    def __init__(self, msg):
        super().__init__(msg)


class FrameNotInitializeError(FrameError):
    '''Frame being used while not being setup'''
    def __init__(self, msg):
        super().__init__(msg)


class FrameAttributeNotFoundError(FrameError):
    '''Try to access Frame with not registered attribute'''
    def __init__(self, msg):
        super().__init__(msg)


class FrameAttributeExistError(FrameError):
    '''Try to register attribute with exist name'''
    def __init__(self, msg):
        super().__init__(msg)


class FrameAttributeNotRegisteredError(FrameError):
    '''Try to setup a Frame without registered any attributes'''
    def __init__(self):
        super().__init__("Frame has no attributes registered.")


class FrameAlreadySetupError(FrameError):
    '''Try to register an attribute after setup'''
    def __init__(self):
        super().__init__("Frame already being setup, cannot register attributes.")


class SnapshotAccessError(FrameError):
    '''Snapshot cannot be wrote'''
    def __init__(self):
        super().__init__("Snapshot cannot be wrote.")


class SnapshotSliceError(FrameError):
    '''Using invalid parameter to query snapshot with slice interface'''
    def __init__(self, msg):
        super().__init__(msg)


# TODO: used to make the function more general, not used now
#ctypedef fused float_or_int:
#    FLOAT
#    INT


# TODO: seems like inline only can speed up only if it not a class function, so we keep a standalone function with lot of parameters here
# any way to simple it?

# function to calculation offset of specified index
# 
# Parameters:
# node_type: type of node, static or dynamic
# node_id: id of node to query
# attribute: registered attribute
# attribute_index: index of registered attribute
# float_row_size: size of each row in float data block
# int_row_size: size of each row in int data block
# static_node_num: static node number that used to calculate offset

cdef inline UINT get_attribute_offset(UINT node_type, UINT node_id, FrameAttribute attribute, UINT attribute_index, UINT float_row_size, UINT int_row_size, UINT static_node_num):
    cdef UINT node_offset = node_type * static_node_num + node_id
    cdef UINT attr_offset = attribute.start_index + attribute_index
    cdef UINT attr_type = attribute.attribute_type

    if attr_type == ATTR_TYPE_FLOAT:
        return node_offset * float_row_size + attr_offset
    else:
        return node_offset * int_row_size + attr_offset      


cdef class Frame:
    '''Frame used to hold attributes for both static and dynamic nodes.

    To initialize a Frame, attributes must to be registered before setup.

    Example:
        Create a simple Frame that with 10 static and 10 dynamic nodes, and attributes like "attr1", "attr2":
            static_node_num = 10
            dynamic_node_num = 10

            # init the Frame object first
            frame = Frame(static_node_num, dynamic_node_num)

            # then register attributes

            # register an attribute named "attr1", its data type is float, can hold 1 value (slot)
            frame.register_attribute("attr1",  FrameAttributeType.FLOAT, 1)

            # register an attribute named "attr2", its data type is int, can hold 2 value (slots)
            frame.register_attribute("attr2", FrameAttributeType.INT, 2)

            # then we can setup the Frame for using
            frame.setup()

            # the frame is ready to accessing now

            # get an attribute (first slot) of a static node that id is 0
            a1 = frame.get_attribute(FrameNodeType.STATIC, 0, "attr1", 0)

            # set an attribute (2nd slot) of a dynamic node that id is 0
            frame.set_attribute(FrameNodeType.DYNAMIC, 0, "attr2", 1, 123)

    Args:
        static_node_num (int): number of static nodes in Frame
        dynamic_node_num (int): number of dynamic nodes in Frame
    '''
    cdef:
        # memory view to hold frame attributes
        FLOAT [:] float_data_block
        INT [:] int_data_block
        INT [:] int_mat_data_block

        UINT static_node_num
        UINT dynamic_node_num

        UINT float_row_size
        UINT int_row_size
        UINT int_mat_row_size

        # we use this mapping to give outside a shorter way to fetch attribute value
        dict attribute_map

        # if the setup method already called, used to avoid call register after setup
        bool is_initialized

    def __cinit__(self, UINT static_node_num, UINT dynamic_node_num):
        self.static_node_num = static_node_num
        self.dynamic_node_num = dynamic_node_num
        self.float_row_size = 0
        self.int_row_size = 0
        self.int_mat_row_size = 0
        self.is_initialized = False

        self.attribute_map = {}

    cpdef setup(self):
        '''Setup the Frame with registered attributes

        Raises:
            FrameAttributeNotRegisteredError: if not registered any attribute
        '''
        if self.is_initialized:
            return

        if len(self.attribute_map) == 0:
            raise FrameAttributeNotRegisteredError()

        self.is_initialized = True

        cdef UINT total_node_num = self.static_node_num + self.dynamic_node_num

        self.float_data_block = np.zeros((total_node_num * self.float_row_size), dtype=FLOAT_DTYPE)
        self.int_data_block = np.zeros((total_node_num * self.int_row_size), dtype=INT_DTYPE)
        self.int_mat_data_block = np.zeros(self.int_mat_row_size, dtype=INT_DTYPE)

    cpdef reset(self):
        '''Reset all the attributes to default value'''
        cdef UINT total_node_num = self.static_node_num + self.dynamic_node_num
        cdef UINT index = 0

        for i in range(total_node_num):
            for j in range(self.float_row_size):
                index = i * self.float_row_size + j
                self.float_data_block[index] = 0.0

            for j in range(self.int_row_size):
                index = i * self.int_row_size + j
                self.int_data_block[index] = 0

        index = 0

        for index in range(self.int_mat_row_size):
            self.int_mat_data_block[index] = 0

    cpdef void register_attribute(self, str name, UINT attribute_type, UINT size, UINT row_num=0, UINT column_num=0):
        '''Register an attribute for nodes in Frame, then can access the new attribute with get/set_attribute methods.

        NOTE: this method should be called before setup method

        Args:
            name (str): name of the attribute 
            attribute_type (FrameAttributeType): data type of attribute, only float and int now
            size (int): how many slots of this attributes can hold
            row_num (int): matrix row number, for matrix attribute only
            colum_num (int): matrix column number, for matrix attribute only

        Raises:
            FrameAttributeExistError: if the name already being registered
        '''
        if self.is_initialized == True:
            return

        if name in self.attribute_map:
            return
            # raise FrameAttributeExistError(f"Attribute name {name} already registered.")

        cdef UINT start_index = 0

        if attribute_type == ATTR_TYPE_FLOAT:
            start_index = self.float_row_size
            self.float_row_size += size
        elif attribute_type == ATTR_TYPE_INT:
            start_index = self.int_row_size
            self.int_row_size += size
        elif attribute_type == ATTR_TYPE_INT_MAT:
            start_index = self.int_mat_row_size
            self.int_mat_row_size += size

        cdef FrameAttribute new_attribute = FrameAttribute(name, attribute_type, size, start_index, row_num, column_num)

        self.attribute_map[name] = new_attribute

    cpdef get_attribute(self, UINT node_type, UINT node_id, str attribute_name, UINT attribute_index):
        '''Get specified attribute value with general way

        Args:
            node_type (FrameNodeType): Frame node type, static or dynamic
            node_id (int): id the the Frame node
            attribute_name (str): name of accessing attribute
            attribute_index (int): index of the attribute slot

        Returns:
            value of specified attribute slot, can be float or int fow now

        Raises:
            FrameAttributeNotFoundError: specified attribute is not registered
        '''
        if attribute_name not in self.attribute_map:
            raise FrameAttributeNotFoundError(f"Attribute {attribute_name} is not registered.")

        cdef UINT offset = get_attribute_offset(node_type, node_id, self.attribute_map[attribute_name], attribute_index, self.float_row_size, self.int_row_size, self.static_node_num)
        cdef FrameAttribute attribute = self.attribute_map[attribute_name]
        cdef UINT attribute_type = attribute.attribute_type

        if attribute_type == ATTR_TYPE_FLOAT:            
            return self.float_data_block[offset]
        elif attribute_type == ATTR_TYPE_INT:
            return self.int_data_block[offset]

    cpdef set_attribute(self, UINT node_type, UINT node_id, str attribute_name, UINT attribute_index, object value):
        '''Set specified attribute value

        Args:
            node_type (FrameNodeType): Frame node type, static or dynamic
            node_id (int): id the the Frame node
            attribute_name (str): name of accessing attribute
            attribute_index (int): index of the attribute slot        
            value (float/int): value to set

        Raises:
            FrameAttributeNotFoundError: specified attribute is not registered
        '''
        if attribute_name not in self.attribute_map:
            raise FrameAttributeNotFoundError(f"attribute {attribute_name} is not registered.")

        cdef UINT offset = get_attribute_offset(node_type, node_id, self.attribute_map[attribute_name], attribute_index, self.float_row_size, self.int_row_size, self.static_node_num)
        cdef FrameAttribute attribute = self.attribute_map[attribute_name]
        cdef UINT attribute_type = attribute.attribute_type

        if attribute_type == ATTR_TYPE_FLOAT:
            self.float_data_block[offset] = value
        elif attribute_type == ATTR_TYPE_INT:
            self.int_data_block[offset] = value

    cpdef set_int_matrix_value(self, str attribute_name, UINT row_index, UINT column_index, INT value):
        '''Set value of specified matrix field

        NOTE: this method may be changed after next version

        Args:
            attribute_name (str): matrix name
            row_index (int): index of the row
            column_index (int): index of the column
            value (int): value to set

        Raises:
            FrameAttributeNotFoundError: if attribute not find

        '''
        if attribute_name not in self.attribute_map:
            raise FrameAttributeNotFoundError(f"attribute {attribute_name} is not registered.")

        cdef FrameAttribute attribute = self.attribute_map[attribute_name]
        cdef UINT attribute_type = attribute.attribute_type
        cdef UINT attribute_size = attribute.size
        cdef UINT offset = attribute.start_index
        cdef UINT row_num = attribute.row_num
        cdef UINT column_num = attribute.column_num

        if attribute_type != ATTR_TYPE_INT_MAT:
            raise FrameInvalidAccessError(f"Attribute {attribute_name} is not a int matrix attribute")

        if row_index >= row_num or column_index > column_num:
            raise FrameInvalidAccessError(f"Length of input value not match attribute")

        self.int_mat_data_block[offset + row_index * column_num + column_index] = value


    cpdef  INT get_int_matrix_value(self, str attribute_name, UINT row_index, UINT column_index):
        '''Get value of specified matrix field

        Args:
            attribute_name (str): matrix name
            row_index (int): index of the row
            column_index (int): index of the column

        Raises:
            FrameAttributeNotFoundError: if attribute not find

        '''
        cdef FrameAttribute attribute = self.attribute_map[attribute_name]
        cdef UINT attribute_type = attribute.attribute_type
        cdef UINT attribute_size = attribute.size
        cdef UINT offset = attribute.start_index
        cdef UINT row_num = attribute.row_num
        cdef UINT column_num = attribute.column_num

        if attribute_type != ATTR_TYPE_INT_MAT:
            raise FrameInvalidAccessError(f"Attribute {attribute_name} is not a int matrix attribute")

        if row_index >= row_num or column_index > column_num:
            raise FrameInvalidAccessError(f"Length of input value not match attribute")

        return self.int_mat_data_block[offset + row_index * column_num + column_index]

    @property
    def static_node_number(self) -> int:
        '''int: Number of static nodes in current Frame'''
        return self.static_node_num

    @property
    def dynamic_node_number(self) -> int:
        '''int: Number of dynamic nodes in current Frame'''
        return self.dynamic_node_num


# more on memory view: https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html


cdef class SnapshotList:
    '''SnapshotList used to hold list of snapshots that taken from Frame object at a certain tick.

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
        SnapshotAccessor _dynamic_nodes
        SnapshotAccessor _static_nodes
        SnapshotMatrixAccessor _matrix

        # each row is a frame of a tick
        FLOAT [:, :] float_data_block
        INT [:, :] int_data_block
        INT [:, :] int_mat_data_block

        # index of latest snapshot
        INT latest_index
        INT max_tick

        # used for quick access to avoid repeatly calculations
        UINT frame_float_block_size
        UINT frame_int_block_size
        UINT frame_float_row_size
        UINT frame_int_row_size
        UINT frame_int_mat_size
        UINT static_node_num
        UINT dynamic_node_num

        # refence to frame attributes, used to map attribute name
        dict attribute_map
    
    def __cinit__(self, Frame frame, UINT max_tick):
        '''Create a new instance of SnapshotList

        Args:
            frame (Frame): frame that need to take the shape, later we can only accept this shape of frame to take snapshot
            max_tick (int): max ticks in this snapshot list
        '''

        self._dynamic_nodes = SnapshotAccessor(FrameNodeType.DYNAMIC, frame.dynamic_node_num, self)
        self._static_nodes = SnapshotAccessor(FrameNodeType.STATIC, frame.static_node_num, self)
        self._matrix = SnapshotMatrixAccessor(self)

        cdef UINT total_node_num = frame.static_node_num + frame.dynamic_node_num

        self.dynamic_node_num = frame.dynamic_node_num
        self.static_node_num = frame.static_node_num
        self.frame_float_row_size = frame.float_row_size
        self.frame_int_row_size = frame.int_row_size
        self.frame_float_block_size = total_node_num * frame.float_row_size
        self.frame_int_block_size = total_node_num * frame.int_row_size
        self.frame_int_mat_size = frame.int_mat_row_size

        self.float_data_block = np.zeros((max_tick, self.frame_float_block_size), dtype=FLOAT_DTYPE)
        self.int_data_block = np.zeros((max_tick, self.frame_int_block_size), dtype=INT_DTYPE)
        self.int_mat_data_block = np.zeros((max_tick, self.frame_int_mat_size), dtype=INT_DTYPE)

        self.max_tick = max_tick
        self.latest_index = -1 # no snapshot 
        self.attribute_map = frame.attribute_map

    def reset(self):
        self.latest_index = -1

    cpdef UINT get_slot_length(self, str attribute_name):
        '''Get slot length of specified attribute

        Args:
            attribute_name (str): name of the attribute

        Returns:
            int: slot length of attribute, 0 if attribute not exist
        '''
        if attribute_name in self.attribute_map:
            return self.attribute_map[attribute_name].size

        return 0

    cpdef void insert_snapshot(self, Frame frame, UINT tick):
        '''Insert a snapshot from specified frame

        Args:
            frame (Frame): frame to take snapshot
            tick (int): tick of current frame
        '''
        self.float_data_block[tick:] = frame.float_data_block
        self.int_data_block[tick:] = frame.int_data_block
        self.int_mat_data_block[tick:] = frame.int_mat_data_block

        self.latest_index = tick

    cpdef get_attribute(self, UINT tick, UINT node_type, UINT node_id, str attribute_name, UINT attribute_index):
        '''
        Get an attribute value from snapshot list at specified tick.

        Example:
            get 1st slot value of attribute "a1" from dynamic node "12" at tick "100"

            attr_value = snapshotlist.get_attribute(100, FrameNodeType.DYNAMIC, 12, "a1", 0)

        Args:
            tick (int): tick of the snapshot
            node_type (FrameNodeType): frame node type to get
            node_id (int): id of the frame node
            attribute_name (str): name of the attribute
            attribute_index (int): slot index of the attribute

        Returns:
            int/float: value

        Raises:
            FrameAttributeNotFoundError: if the attribute name not being registered
        '''
        
        if attribute_name not in self.attribute_map:
            raise FrameAttributeNotFoundError(f"cannot find attribute {attribute_name} in current frame.")

        cdef FrameAttribute attribute = self.attribute_map[attribute_name]
        cdef UINT node_offset = node_type* self.static_node_num + node_id
        cdef UINT attribute_offset = attribute.start_index + attribute_index
        cdef attribute_type = attribute.attribute_type


        if attribute_type == ATTR_TYPE_FLOAT:            
            return self.float_data_block[tick, node_offset * self.frame_float_row_size + attribute_offset]
        elif attribute_type == ATTR_TYPE_INT:
            return self.int_data_block[tick, node_offset * self.frame_int_row_size + attribute_offset]

        raise FrameInvalidAccessError("invalid type to access")

    cpdef np.ndarray get_matrix(self, list ticks, str attribute_name):
        '''Get a matrix attribute for specified ticks.

        NOTE: it is recommended to use slice method (snapshot_list.matrix) to query.

        Args:
            ticks (list): list of ticks to query
            attribute_name (str): name of attribute name to query
        '''
        if attribute_name not in self.attribute_map:
            raise FrameAttributeNotFoundError(f"cannot find attribute {attribute_name} in current frame.")

        cdef FrameAttribute attribute = self.attribute_map[attribute_name]
        cdef attribute_type = attribute.attribute_type

        if attribute_type != ATTR_TYPE_INT_MAT:
            raise FrameInvalidAccessError(f"Attribute is not a matrix, please use get_attribute to retrieve value")

        cdef ticks_length = len(ticks)
        cdef attribute_size = attribute.size
        cdef UINT start = attribute.start_index
        cdef UINT end = start +attribute_size
        cdef np.ndarray ret = np.zeros((ticks_length, attribute_size), dtype=INT_DTYPE)
        cdef INT [:, :] ret_view = ret
        cdef UINT i = 0
        cdef UINT j = 0
        cdef INT tick = 0
        cdef INT latest_index = self.latest_index

        for i in range(ticks_length):
            tick = ticks[i]

            # skip tick if not valid
            if tick < 0 or tick > latest_index:
                continue

            for j in range(start, end):
                ret_view[i, j-start] = self.int_mat_data_block[tick, j]

        return ret

    # TODO: can we optimize this to make it faster?
    cpdef np.ndarray get_attributes(self, UINT node_type, list ticks, list node_ids, list attribute_names, list attribute_indices):
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
            node_type (FrameNodeType): type of resource node, static or dynamic
            ticks (list[int]): list of tick to query, if the tick not available, then related value will be 0
            node_ids (list[int]): list of node id, if the id not exist, then the related value will be 0
            attribute_names (list[str]): attribute names to query, if the attribute not exist, then the related value will be 0
            attribute_indices (list[int]): slots of attributes to query, if the index is large than registered size, then related value will be set 0 for it
        
        Returns:
            np.ndarray: numpy array (dim=1, size=len(ticks) * len(node_ids) * len(attribute_names) * len(attribute_indices)) with result
        '''
        cdef UINT ridx = 0
        cdef INT tick = 0
        cdef UINT node_offset = 0
        cdef UINT attr_offset = 0
        cdef UINT attr_type = ATTR_TYPE_FLOAT
        cdef UINT static_node_num = self.static_node_num
        cdef UINT float_row_size = self.frame_float_row_size
        cdef UINT int_row_size = self.frame_int_row_size
        cdef UINT index = 0

        cdef UINT max_node_num = self.static_node_num
        cdef INT latest_index = self.latest_index  

        # length of each parameter
        cdef UINT ticks_len = len(ticks)
        cdef UINT ids_len = len(node_ids)
        cdef UINT attrs_len = len(attribute_names)
        cdef UINT index_len = len(attribute_indices)

        # result to return
        cdef np.ndarray result = np.zeros(ticks_len * ids_len * attrs_len * index_len, dtype=FLOAT_DTYPE)

        cdef FLOAT [:] result_view = result

        if node_type == NODE_TYPE_DYNAMIC:
            max_node_num = self.dynamic_node_num

        for tick in ticks:
            # skip tick if not valid
            if tick < 0 or tick > latest_index:
                ridx += ids_len * attrs_len * index_len

                continue
                
            for node_id in node_ids:
                # skip node if is not valid
                if node_id < 0 or node_id >= max_node_num:
                    ridx += attrs_len * index_len

                    continue

                for attr_name in attribute_names:
                    for attr_index in attribute_indices:
                        # fill with value if it is valid attribute and index
                        if attr_name in self.attribute_map:
                            attr = self.attribute_map[attr_name]
                            attr_type = attr.attribute_type

                            if attr_type == ATTR_TYPE_INT_MAT:
                                raise FrameInvalidAccessError(f"{attr_name} is not support in get_attributes interface")

                            if attr_index < attr.size:
                                node_offset = node_type * static_node_num + node_id
                                attr_offset = attr.start_index + attr_index

                                if attr_type == ATTR_TYPE_FLOAT:
                                    index = node_offset * float_row_size + attr_offset 

                                    v = self.float_data_block[tick, index]
                                elif attr_type == ATTR_TYPE_INT:
                                    index = node_offset * int_row_size + attr_offset   

                                    v = self.int_data_block[tick, index]

                                result_view[ridx] = <FLOAT>v

                        ridx += 1
        
        return result

    def __len__(self):
        return self.max_tick

    @property
    def dynamic_node_number(self) -> int:
        '''int: Dynamic node number in each snapshot'''
        return self.dynamic_node_num

    @property
    def static_node_number(self) -> int:
        '''int: Static node number in each snapshot'''
        return self.static_node_num

    @property
    def attributes(self):
        '''Get list of the attributes information in current snapshot

        Returns:
            list: A list of attribute details
        '''
        result = []

        for attr_name, attr in self.attribute_map.items():
            result.append({"name": attr_name, "slot length": attr.size})

        return result

    @property
    def dynamic_nodes(self) -> SnapshotAccessor:
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
        return self._dynamic_nodes

    @property
    def static_nodes(self) -> SnapshotAccessor:
        '''Same as dynamic_nodes, but for static nodes'''
        return self._static_nodes

    @property
    def matrix(self) -> np.ndarray:
        '''Slice interface to access matrix attributes

        The slice like [tick: name]

        tick: tick to query, can be a list
        name: name of the matrix to query

        Examples:
            # query a matrix at tick 0
            snapshot_list.matrix[0: "m1"]

            # query matrix at tick 0 and 1
            snapshot_list.matrix[[0, 1]: "m1"]
        '''
        return self._matrix


cdef class SnapshotAccessor:
    '''Easy attributes helper class for snapshot list'''
    cdef:
        UINT node_type
        UINT node_number
        SnapshotList snapshot_list

    def __cinit__(self, UINT node_type, UINT node_number, SnapshotList snapshot_list):
        self.node_type = node_type
        self.snapshot_list = snapshot_list
        self.node_number = node_number

    def __getitem__(self, slice item):
        '''Query states from snapshot'''
        cdef list ticks
        cdef list id_list
        cdef list attribute_names
        cdef list attribute_indices

        # if ticks and id list, they can be empty, then means take all
        if item.start is None:
            ticks = [i for i in range(len(self))]
        else:
            # convenient way if user need to query only one attribute or one slot
            if type(item.start) is not list:
                ticks = [item.start]
            else:
                ticks = item.start

        if item.stop is None:
            id_list = [i for i in range(self.node_number)]
        else:
            if type(item.stop) is not list:
                id_list = [item.stop]
            else:
                id_list = item.stop

        if item.step is None or len(item.step) < 2:
            raise SnapshotSliceError("Invalid attributes to slice, please provide a tuple with length == 2, like (attribute_names, attributes_indices)")

        cdef tuple attributes = item.step

        if type(attributes[0]) is not list:
            attribute_names = [attributes[0]]
        else:
            attribute_names = attributes[0]

        if type(attributes[1]) is not list:
            attribute_indices = [attributes[1]]
        else:
            attribute_indices = attributes[1]

        return self.snapshot_list.get_attributes(self.node_type, ticks, id_list, attribute_names, attribute_indices)

    def __len__(self):
        return len(self.snapshot_list)

    def __setitem__(self, item, value):
        raise SnapshotAccessError("Cannot set value of snapshot list")


cdef class SnapshotMatrixAccessor:
    '''Wrapper to make it easy to access matrix of snapshot'''
    cdef:
        SnapshotList snapshot_list

    def __cinit__(self, SnapshotList snapshot_list):
        self.snapshot_list = snapshot_list

    def __getitem__(self, slice item):
        cdef list ticks
        cdef str attr_name = item.stop

        if type(item.start) is not list:
            ticks = [item.start]
        else:
            ticks = item.start

        return self.snapshot_list.get_matrix(ticks, attr_name)