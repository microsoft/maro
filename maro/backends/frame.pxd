# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3

from cpython cimport bool

from maro.backends.backend cimport BackendAbc, SnapshotListAbc


cdef class SnapshotList:
    """List of frame snapshot for ticks (frame index), usually used to query states.
    

    SnapshotList is read-only for out-side of simulator, it provides a slice interface to query states for nodes at specified ticks (frame index).


    Same as frame, snapshot list is composed with serveral nodes, you should get snapshot list for a node with node name, before querying,
    then use the slice interface to query.


    Slice interface accept 3 parameters: tick/tick_list (frame index), node_index/node_index_list, attribute_name/attribute_name_list,
    tick and node_index can be empty, then means all the ticks (frame index) or all the nodes.


    When querying, all the slot of specified attribute will be returned.
    The querying result is a 1 dim numpy array, and grouped like: [[node[attr] for node in nodes] attr for attr in attributes] * len(ticks)


    NOTE:
        Slice interface returns a 1-dim numpy array, you may need to reshape it as your requirement. Also the attribute must defined in specified node,
        or will cause error.

        
        Invalid tick (frame index) will be padding in return value, that means all the value for that tick (frame index) will be 0, and will not cause error.


    Examples:

        .. code-block:: python

            # suppose it contains 'yournodes' and 'mynodes' in definition
            frame = MyFrame()

            # get snapshots of 'mynodes' node
            my_snapshots = frame.snapshots["mynodes"]

            # query attributes states from snapshot list
            # get values of all 'mynodes' node at all the tick
            my_value_at_all_tick = my_snapshot[::"value"]

            # get values of all 'mynodes' node at 1st tick
            my_value_at_1st_tick = my_snapshot[0::"value"]

            # get values of all 'mynodes' node at [0, 2, 3] ticks
            my_values = my_snapshot[(0, 2, 3)::"value"]

            # get all value of 1st 'mynodes' node at 1st tick
            my_values = my_snapshot[0:0:"value"]
    
    """
    cdef:
        SnapshotListAbc _snapshots

        dict _nodes_dict


cdef class FrameBase:
    """Base object used to define frame in backend, any frame that need to be hosted in backend should inherit from this.
    Usually a frame is composed with serveral nodes (NodeBase), a snapshot list if enabled.

    .. code-block:: python

        # normal frame definition
        class MyFrame(FrameBase):
            # assuming we have 2 nodes definition with NodeBase (MyNode, YourNodes)
            mynodes = FrameNode(MyNode, 10)
            yournodes = FrameNode(YourNode, 12)

            def __init__(self, enable_snapshot:bool=True, snapshot_number: int = 10):
                super().__init__(self, enable_snapshot, total_snapshots=snapshot_number)

    
    The snapshot list is used to hold snapshot of current frame at specified point (tick or frame index), it can be
    configured that how many snapshots should be kept in memory, latest snapshot will over-write oldest one if reach the limitation, this is
    useful when the memory is not enough.


    When defining a frame, number of node must be specified, this may not suitable for all the case, such as the node number
    is dynamic generated after initializing, this is a workaround to fix this that use function wrapper to generate frame definition
    at runtime.


    .. code-block:: python

        # using function to wrap the frame definition to support dynamic node number
        def gen_frame_definition(my_node_number: int):
            class MyDynamicFrame(FrameBase):
                mynodes = FrameNode(MyNode, my_node_number)
                yourndes = FrameNode(YourNode, 12)

                def __init__(self):
                    super().__init__(self, True, total_snapshots=10)

            # this is our final frame definition
            return MyDynamicFrame


    After initializing, frame will generate instance list for all the nodes, these list can be accessed by their definition name,
    each node instance will be assigned an index attribute (0 based) for later quering.

    
    .. code-block:: python
        
        frame = MyFrame()

        # get instance list of MyNode
        my_nodes_list = frame.mynodes
        your_nodes_list = frame.yournodes

        for mnode in my_node_list:
            print(mnode.index) # 0 - len(my_node_list)

    
    Args:
        enable_snapshot (bool): if enable snapshot list to keep frame snapshot at specified point, default False
        total_snapshots (int): total snapshots number in memory
        options (dict): additional options, reserved for later using

    
    Attributes:
        snapshots (SnapshotList): property to access snapshot list, readonly, see SnapshotList for details.
    """
    cdef:
        BackendAbc _backend

        SnapshotList _snapshot_list

        # enable dynamic fields
        dict __dict__
        

    cpdef void reset(self) except *

    cpdef void take_snapshot(self, int tick) except *

    cpdef void enable_history(self, str path) except *

    cdef void _setup_backend(self, bool enable_snapshot, int total_snapshot, dict options) except *


cdef class FrameNode:
    """Helper class used to define a node in frame, specified node class and number.
    

    Usually use with FrameBase, see FrameBase for details.
    
    
    Args:
        node_cls(type): class type of node definition inherit from NodeBase
        number(int): the number of this node in Frame
    """
    cdef:
        public type _node_cls

        public int _number


cdef class NodeBase:
    """Helper class used to define a node with attributes, any node that need to be hosted in backend should inherit from this class.
    A node is composed with serveral attributes that defined with NodeAttribute class with data type and slot number. 
        
        
    A node must have a name that used to query states in snapshot list, this name is specified via @node decorator.


    NOTE:
        A node definition must decorated with @node decorator to specified a name, or will cause error.


    To add an attribute in node definition, just add class variable that type is NodeAttribute to your node class.


    .. code-block:: python

        # node name in Frame, we use this name to query from snapshot list
        @node("my nodes")
        class MyNode(NodeBase): # node class we used in business engine
            # attribute name, and its data type
            my_int_attr = NodeAttribute("i") # int attribute with 1 value
            my_float_array_attr = NodeAttribute("f", 2) # a fixed size float array


    Same as frame definition, if you need dynamic slot number, you use a function that wrap a node defnition too.


    NOTE:
        Do not create instance of nodes by yourself, it will cause error.


    .. code-block:: python

        def gen_my_node_definition(float_attr_number: int):
            @node("my nodes")
            class MyNode(NodeBase):
                my_int_attr = NodeAttribute("i")
                my_float_array_attr = NodeAttribute("f", float_attr_number) 

            return MyNode


    After frame initialzing, we can access these attribute via frame node instance list for this node type.
    For attributes which slot number is 1, you can access it as normal python object property.


    .. code-block:: python

        f = MyFrame()

        # get MyNode instance list
        my_nodes = f.mynodes

        # 1st MyNode
        mnode_1 = my_nodes[0]

        # access predefined attributes
        print(mnode_1.my_int_attr)


    For attributes that slot number greater than 1, we provided a slice interface to make it easy to access.


    NOTE:
        Accessing attributes that slot number greater than 1 should alway use slice interface, cannot assign value to them 
        directly, this will cause error.


    .. code-block:: python

        # accessing by slot index to get a value at that slot
        print(my_nodes.my_float_array_attr[0])

        # accessing with range of slot index, will return values at those slots
        print(my_nodes.my_float_array_attr[(0, 1)])

        # get all values at all slots of this attribute
        print(my_nodes.my_float_array_attr[:])

        # also you can use same way to assign values
        my_nodes.my_float_array_attr[0] = 1.1
        my_nodes.my_float_array_attr[(0, 1)] = (0.1, 0.2)

    """
    cdef:
        # index of current node in frame memory,
        # all the node/frame operation will base on this property, so user should create a mapping that
        # map the business model id/name to node index
        int _index

        BackendAbc _backend

        # enable dynamic attributes
        dict __dict__

    # set up the node for using with frame, and index
    # this is called by Frame after the instance is initialized
    cdef void setup(self, BackendAbc backend, int index) except *

    # internal functions, will be called after Frame's setup, used to bind attributes to instance
    cdef void _bind_attributes(self) except *


cdef class NodeAttribute:
    """Helper class used to declare an attribute in node that inherit from NodeBase.
    

    Currently we only support these data types: 'i', 'i2', 'i4', 'i8', 'f', 'd'


    Args:
        dtype(str): type of this attribute, it support following data types.
        slots(int): if this number greater than 1, then it will be treat as an array, this will be the array size.
    """
    cdef:
        # data type of attribute, same as numpy string dtype
        public str _dtype

        # array size of tis attribute
        public int _slot_number