
Data Model
==========

The data model of MARO provides a declarative interface. We choose Python as
the frontend language for saving development cost, and we take C as
the backend language for improving the execution reference. What's more,
the backend store is a pluggable design, user can choose different backend
implementation based on their real performance requirement and device limitation.

Currently there are two data model backend implementation: static and dynamic.
Static implementation used Numpy as its data store, do not support dynamic
attribute length, the advance of this version is that its memory size is same as its
declaration.
Dynamic implementation is hand-craft c++.
It supports dynamic attribute (list) which will take more memory than the static implementation
but is faster for querying snapshot states and accessing attributes.

Key Concepts
------------

.. image:: ../images/simulator/key_concepts.svg
   :target: ../images/simulator/key_concepts.svg
   :alt: Key Concepts
   :width: 220

As shown in the figure above, there are some key concepts in the data model:

* **Node** is the abstraction of the resource holder, which is usually the major
  business instance of the scenario (i.e. vessels and ports in CIM scenario). A
  node usually has various attributes to present the business nature.
* **(Slot) Attribute** is the abstraction of business properties for the
  scenarios-specific resource holder (node). The attributes of a node could be
  declared as different data types based on the real requirements. Furthermore,
  for each attribute, a ``slot`` feature is provided to support the fixed-size array.
  The ``slot`` number can indicate the attribute values (e.g. the three different
  container types in CIM scenario) or the detailed categories (e.g. the ten specific
  products in the `Use Case <#use-case>`_ below). By default, the ``slot`` value is one.
  As for the dynamic backend implementation, an attribute can be marked as is_list or is_const to identify
  it is a list attribute or a const attribute respectively.
  A list attribute's default slot number is 0, and can be increased as demand, max number is 2^32.
  A const attribute is designed for the value that will not change after initialization,
  e.g. the capacity of a port/station. The value is shared between frames and will not be copied
  when taking a snapshot.
* **Frame** is the collection of all nodes in the environment. The historical frames
  present the aggregated state of the environment during a specific period, while
  the current frame hosts the latest state of the environment at the current time point.
* **Snapshot List** is the dumped frames based on a pre-defined resolution.
  It captures the aggregated changes of the environment between the dump points.

Use Case
--------

* Below is the declaration of a retail frame, which includes warehouse and store nodes.

  .. code-block:: python

    from maro.backends.backend import AttributeType
    from maro.backends.frame import node, NodeAttribute, NodeBase, FrameNode, FrameBase

    TOTAL_PRODUCT_CATEGORIES = 10
    TOTAL_STORES = 8
    TOTAL_WAREHOUSES = 2
    TOTAL_SNAPSHOT = 100


    @node("warehouse")
    class Warehouse(NodeBase):
        inventories = NodeAttribute(AttributeType.Int, TOTAL_PRODUCT_CATEGORIES)
        shortages = NodeAttribute(AttributeType.Int, TOTAL_PRODUCT_CATEGORIES)

        def __init__(self):
            self._init_inventories = [100 * (i + 1) for i in range(TOTAL_PRODUCT_CATEGORIES)]
            self._init_shortages = [0] * TOTAL_PRODUCT_CATEGORIES

        def reset(self):
            self.inventories[:] = self._init_inventories
            self.shortages[:] = self._init_shortages


    @node("store")
    class Store(NodeBase):
        inventories = NodeAttribute(AttributeType.Int, TOTAL_PRODUCT_CATEGORIES)
        shortages = NodeAttribute(AttributeType.Int, TOTAL_PRODUCT_CATEGORIES)
        sales = NodeAttribute(AttributeType.Int, TOTAL_PRODUCT_CATEGORIES)

        def __init__(self):
            self._init_inventories = [10 * (i + 1) for i in range(TOTAL_PRODUCT_CATEGORIES)]
            self._init_shortages = [0] * TOTAL_PRODUCT_CATEGORIES
            self._init_sales = [0] * TOTAL_PRODUCT_CATEGORIES

        def reset(self):
            self.inventories[:] = self._init_inventories
            self.shortages[:] = self._init_shortages
            self.sales[:] = self._init_sales


    class RetailFrame(FrameBase):
        warehouses = FrameNode(Warehouse, TOTAL_WAREHOUSES)
        stores = FrameNode(Store, TOTAL_STORES)

        def __init__(self):
            # If your actual frame number was more than the total snapshot number, the old snapshots would be rolling replaced.
            # You can select a backend implementation that will fit your requirement.
            super().__init__(enable_snapshot=True, total_snapshot=TOTAL_SNAPSHOT, backend_name="static/dynamic")

* The operations on the retail frame.

  .. code-block:: python

    retail_frame = RetailFrame()

    # Fulfill the initialization values to the backend memory.
    for store in retail_frame.stores:
        store.reset()

    # Fulfill the initialization values to the backend memory.
    for warehouse in retail_frame.warehouses:
        warehouse.reset()

    # Take a snapshot of the first tick frame.
    retail_frame.take_snapshot(0)
    snapshot_list = retail_frame.snapshots
    print(f"Max snapshot list capacity: {len(snapshot_list)}")

    # Query sales, inventory information of all stores at first tick, len(snapshot_list["store"]) equals to TOTAL_STORES.
    all_stores_info = snapshot_list["store"][0::["sales", "inventories"]].reshape(TOTAL_STORES, -1)
    print(f"All stores information at first tick (numpy array): {all_stores_info}")

    # Query shortage information of first store at first tick.
    first_store_shortage = snapshot_list["store"][0:0:"shortages"]
    print(f"First store shortages at first tick (numpy array): {first_store_shortage}")

    # Query inventory information of all warehouses at first tick, len(snapshot_list["warehouse"]) equals to TOTAL_WAREHOUSES.
    all_warehouses_info = snapshot_list["warehouse"][0::"inventories"].reshape(TOTAL_WAREHOUSES, -1)
    print(f"All warehouses information at first tick (numpy array): {all_warehouses_info}")

    # Add fake shortages to first store.
    retail_frame.stores[0].shortages[:] = [i + 1 for i in range(TOTAL_PRODUCT_CATEGORIES)]
    retail_frame.take_snapshot(1)

    # Query shortage information of first and second store at first and second tick.
    store_shortage_history = snapshot_list["store"][[0, 1]: [0, 1]: "shortages"].reshape(2, -1)
    print(f"First and second store shortage history at the first and second tick (numpy array): {store_shortage_history}")

Supported Attribute Data Type
-----------------------------

All supported data types for the attribute of the node:

.. list-table::
   :widths: 25 25 60
   :header-rows: 1

   * - Attribute Data Type
     - C Type
     - Range
   * - Attribute.Byte
     - char
     - -128 .. 127
   * - Attribute.UByte
     - unsigned char
     - 0 .. 255
   * - Attribute.Short (i2)
     - short
     - -32,768 .. 32,767
   * - Attribute.UShort
     - unsigned short
     - 0 .. 65,535
   * - Attribute.Int (i4)
     - int32_t
     - -2,147,483,648 .. 2,147,483,647
   * - Attribute.UInt (i4)
     - uint32_t
     - 0 .. 4,294,967,295
   * - Attribute.Long (i8)
     - int64_t
     - -9,223,372,036,854,775,808 .. 9,223,372,036,854,775,807
   * - Attribute.ULong (i8)
     - uint64_t
     - 0 .. 18,446,744,073,709,551,615
   * - Attribute.Float (f)
     - float
     - -3.4E38 .. 3.4E38
   * - Attribute.Double (d)
     - double
     - -1.7E308 .. 1.7E308

Advanced Features
-----------------

For better data access, we also provide some advanced features, including:

* **Attribute value change handler**\ : It is a hook function for the value change
  event on a specific attribute. The member function with the
  ``_on_{attribute_name}_changed`` naming pattern will be automatically invoked when
  the related attribute value changed. Below is the example code:

  .. code-block:: python

    from maro.backends.frame import node, NodeBase, NodeAttribute

    @node("test_node")
    class TestNode(NodeBase):
        test_attribute = NodeAttribute("i")

        def _on_test_attribute_changed(self, value: int):
            pass

* **Snapshot list slicing**\ : It provides a slicing interface for querying
  temporal (frame), spatial (node), intra-node (attribute) information. Both a
  single index and an index list are supported for querying specific frame(s),
  node(s), and attribute(s), while the empty means querying all. The return value
  is a flattened 1-dimension NumPy array, which aligns with the slicing order as below:

  .. image:: ../images/simulator/snapshot_list_slicing.svg
    :target: ../images/simulator/snapshot_list_slicing.svg
    :alt: Snapshot List Slicing

  .. code-block:: python

    snapshot_list = env.snapshot_list

    # Get max size of snapshots (in memory).
    print(f"Max snapshot size: {len(snapshot_list)}")

    # Get snapshots of a specific node type.
    test_nodes_snapshots = snapshot_list["test_nodes"]

    # Get node instance amount.
    print(f"Number of test_nodes in the frame: {len(test_nodes_snapshots)}")

    # Query one attribute on all frames and nodes.
    states = test_nodes_snapshots[::"int_attribute"]

    # Query two attributes on all frames and nodes.
    states = test_nodes_snapshots[::["int_attribute", "float_attribute"]]

    # Query one attribute on all frame and the first node.
    states = test_nodes_snapshots[:0:"int_attribute"]

    # Query attribute by node index list.
    states = test_nodes_snapshots[:[0, 1, 2]:"int_attribute"]

    # Query one attribute on the first frame and the first node.
    states = test_nodes_snapshots[0:0:"int_attribute"]

    # Query attribute by frame index list.
    states = test_nodes_snapshots[[0, 1, 2]: 0: "int_attribute"]

    # The querying states is different between static and dynamic implementation
    # Static implementation will return a 1-dim numpy array, as the shape is known according to the parameters.
    # Dynamic implementation will return a 4-dim numpy array, that shape is (ticks, node_indices, attributes, slots).
    # Usually we can just flatten the state from dynamic implementation, then it will be same as static implementation,
    # except for list attributes.
    # List attribute only support one tick, one node index and one attribute name to query, cannot mix with normal attributes
    states = test_nodes_snapshots[0: 0: "list_attribute"]

    # Also with dynamic implementation, we can get the const attributes which is shared between snapshot list, even without
    # any snapshot (need to provided one tick for padding).
    states = test_nodes_snapshots[0: [0, 1]: ["const_attribute", "const_attribute_2"]]



States in built-in scenarios' snapshot list
-------------------------------------------

.. TODO: move to environment part?

Currently there are 3 ways to expose states in built-in scenarios:

Summary
~~~~~~~~~~~

Summary(env.summary) is used to expose static states to outside, it provide 3 items by default:
node_mapping, node_detail and event payload.

The "node_mapping" item usually contains node name and related index, but the structure may be different
for different scenario.

The "node_detail" usually used to expose node definitions, like node name, attribute name and slot number,
this is useful if you want to know what attributes are support for a scenario.

The "event_payload" used show that payload attributes of event in scenario, like "RETURN_FULL" event in
CIM scenario, it contains "src_port_idx", "dest_port_idx" and "quantity".

Metrics
~~~~~~~

Metrics(env.metrics) is designed that used to expose raw states of reward since we have removed reward
support in v0.2 version, and it also can be used to export states that not supported by snapshot list, like dictionary or complex
structures. Currently there are 2 ways to get the metrics from environment: env.metrics, or 1st result from env.step.

This metrics usually is a dictionary with several keys, but this is determined by business engine.

Snapshot_list
~~~~~~~~~~~~~

Snapshot list is the history of nodes (or data model) for a scenario, it only support numberic data types now.
It supported slicing query with a numpy array, so it support batch operations, make it much faster than
using raw python objects.

Nodes and attributes may different for different scenarios, following we will introduce about those in
built-in scenarios.

NOTE:
Per tick state means that the attribute value will be reset to 0 after each step.

CIM
---

Default settings for snapshot list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Snapshot resolution: 1


Max snapshot number: same as durations

Nodes and attributes in scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In CIM scenario, there are 3 node types:


port
++++

capacity
********

type: int
slots: 1

The capacity of port for stocking containers.

empty
*****

type: int
slots: 1

Empty container volume on the port.

full
****

type: int
slots: 1

Laden container volume on the port.

on_shipper
**********

type: int
slots: 1

Empty containers, which are released to the shipper.

on_consignee
************

type: int
slots: 1

Laden containers, which are delivered to the consignee.

shortage
********

type: int
slots: 1

Per tick state. Shortage of empty container at current tick.

acc_storage
***********

type: int
slots: 1

Accumulated shortage number to the current tick.

booking
*******

type: int
slots: 1

Per tick state. Order booking number of a port at the current tick.

acc_booking
***********

type: int
slots: 1

Accumulated order booking number of a port to the current tick.

fulfillment
***********

type: int
slots: 1

Fulfilled order number of a port at the current tick.

acc_fulfillment
***************

type: int
slots: 1

Accumulated fulfilled order number of a port to the current tick.

transfer_cost
*************

type: float
slots: 1

Cost of transferring container, which also covers loading and discharging cost.

vessel
++++++

capacity
********

type: int
slots: 1

The capacity of vessel for transferring containers.

NOTE:
This attribute is ignored in current implementation.

empty
*****

type: int
slots: 1

Empty container volume on the vessel.

full
****

type: int
slots: 1

Laden container volume on the vessel.

remaining_space
***************

type: int
slots: 1

Remaining space of the vessel.

early_discharge
***************

type: int
slots: 1

Discharged empty container number for loading laden containers.

route_idx
*********

type: int
slots: 1

Which route current vessel belongs to.

last_loc_idx
************

type: int
slots: 1

Last stop port index in route, it is used to identify where is current vessel.

next_loc_idx
************

type: int
slots: 1

Next stop port index in route, it is used to identify where is current vessel.

past_stop_list
**************

type: int
slots: dynamic

NOTE:
This and following attribute are special, that its slot number is determined by configuration,
but different with a list attribute, its slot number is fixed at runtime.

Stop indices that we have stopped in the past.

past_stop_tick_list
*******************

type: int
slots: dynamic

Ticks that we stopped at the port in the past.

future_stop_list
****************

type: int
slots: dynamic

Stop indices that we will stop in the future.

future_stop_tick_list
*********************

type: int
slots: dynamic

Ticks that we will stop in the future.

matrices
++++++++

Matrices node is used to store big matrix for ports, vessels and containers.

full_on_ports
*************

type: int
slots: port number * port number

Distribution of full from port to port.

full_on_vessels
***************

type: int
slots: vessel number * port number

Distribution of full from vessel to port.

vessel_plans
************

type: int
slots: vessel number * port number

Planed route info for vessels.

How to
~~~~~~

How to use the matrix(s)
++++++++++++++++++++++++

Matrix is special that it only have one instance (index 0), and the value is saved as a flat 1 dim array, we can reshape it after querying.

.. code-block:: python

  # assuming that we want to use full_on_ports attribute.

  tick = 0

  # we can get the instance number of a node by calling the len method
  port_number = len(env.snapshot_list["port"])

  # this is a 1 dim numpy array
  full_on_ports = env.snapshot_list["matrices"][tick::"full_on_ports"]

  # reshape it, then this is a 2 dim array that from port to port.
  full_on_ports = full_on_ports.reshape(port_number, port_number)

Citi-Bike
---------

Default settings for snapshot list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Snapshot resolution: 60


Max snapshot number: same as durations

Nodes and attributes in scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

station
+++++++

bikes
*****

type: int
slots: 1

How many bikes avaiable in current station.

shortage
********

type: int
slots: 1

Per tick state. Lack number of bikes in current station.

trip_requirement
****************

type: int
slots: 1

Per tick states. How many requirements in current station.

fulfillment
***********

type: int
slots: 1

How many requirement is fit in current station.

capacity
********

type: int
slots: 1

Max number of bikes this station can take.

id
+++

type: int
slots: 1

Id of current station.

weekday
*******

type: short
slots: 1

Weekday at current tick.

temperature
***********

type: short
slots: 1

Temperature at current tick.

weather
*******

type: short
slots: 1

Weather at current tick.

0: sunny, 1: rainy, 2: snowy， 3: sleet.

holiday
*******

type: short
slots: 1

If it is holidy at current tick.

0: holiday, 1: not holiday

extra_cost
**********

type: int
slots: 1

Cost after we reach the capacity after executing action, we have to move extra bikes
to other stations.

transfer_cost
*************

type: int
slots: 1

Cost to execute action to transfer bikes to other station.

failed_return
*************

type: int
slots: 1

Per tick state. How many bikes failed to return to current station.

min_bikes
*********

type: int
slots: 1

Min bikes number in a frame.

matrices
++++++++

trips_adj
*********

type: int
slots: station number * station number

Used to store trip requirement number between 2 stations.


VM-scheduling
-------------

Default settings for snapshot list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Snapshot resolution: 1


Max snapshot number: same as durations

Nodes and attributes in scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cluster
+++++++

id
***

type: short
slots: 1

Id of the cluster.

region_id
*********

type: short
slots: 1

Region is of current cluster.

data_center_id
**************

type: short
slots: 1

Data center id of current cluster.

total_machine_num
******************

type: int
slots: 1

Total number of machines in the cluster.

empty_machine_num
******************

type: int
slots: 1

The number of empty machines in this cluster. A empty machine means that its allocated CPU cores are 0.

data_centers
++++++++++++

id
***

type: short
slots: 1

Id of current data center.

region_id
*********

type: short
slots: 1

Region id of current data center.

zone_id
*******

type: short
slots: 1

Zone id of current data center.

total_machine_num
*****************

type: int
slots: 1

Total number of machine in current data center.

empty_machine_num
*****************

type: int
slots: 1

The number of empty machines in current data center.

pms
+++

Physical machine node.

id
***

type: int
slots: 1

Id of current machine.

cpu_cores_capacity
******************

type: short
slots: 1

Max number of cpu core can be used for current machine.

memory_capacity
***************

type: short
slots: 1

Max number of memory can be used for current machine.

pm_type
*******

type: short
slots: 1

Type of current machine.

cpu_cores_allocated
*******************

type: short
slots: 1

How many cpu core is allocated.

memory_allocated
****************

type: short
slots: 1

How many memory is allocated.

cpu_utilization
***************

type: float
slots: 1

CPU utilization of current machine.

energy_consumption
******************

type: float
slots: 1

Energy consumption of current machine.

oversubscribable
****************

type: short
slots: 1

Physical machine type: non-oversubscribable is -1, empty: 0, oversubscribable is 1.

region_id
*********

type: short
slots: 1

Region id of current machine.

zone_id
*******

type: short
slots: 1

Zone id of current machine.

data_center_id
**************

type: short
slots: 1

Data center id of current machine.

cluster_id
**********

type: short
slots: 1

Cluster id of current machine.

rack_id
*******

type: short
slots: 1

Rack id of current machine.

Rack
++++

id
***

type: int
slots: 1

Id of current rack.

region_id
*********

type: short
slots: 1

Region id of current rack.

zone_id
*******

type: short
slots: 1

Zone id of current rack.

data_center_id
**************

type: short
slots: 1

Data center id of current rack.

cluster_id
**********

type: short
slots: 1

Cluster id of current rack.

total_machine_num
*****************

type: int
slots: 1

Total number of machines on this rack.

empty_machine_num
*****************

type: int
slots: 1

Number of machines that not in use on this rack.

regions
+++++++

id
***

type: short
slots: 1

Id of curent region.

total_machine_num
*****************

type: int
slots: 1

Total number of machines in this region.

empty_machine_num
*****************

type: int
slots: 1

Number of machines that not in use in this region.

zones
+++++

id
***

type: short
slots: 1

Id of this zone.

total_machine_num
*****************

type: int
slots: 1

Total number of machines in this zone.

empty_machine_num
*****************

type: int
slots: 1

Number of machines that not in use in this zone.
