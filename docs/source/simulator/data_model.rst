Data Model
===========

.. image:: ../images/graph.png

As shown in the figure above, data in resource optimization scenarios can be abstracted as graph structures.
Therefore, we employ the spatial-temporal graph as the underlying data model abstraction.

Nodes
-----

Each node in the graph represents a resource carrier.
They can be divided into two types according to their physical properties:

+------------------------+-----------------------------------------------------------------------------------------------------------------+
| Static Resource Nodes  | It's the abstraction for resource repositories, which usually DON'T CHANGE the location in the real world.      |
|                        |                                                                                                                 |
|                        | Examples: container depots of the terminal, parking stations of sharing bicycles.                               |
+------------------------+-----------------------------------------------------------------------------------------------------------------+
| Dynamic Resource Nodes | It's the abstraction for resource containers, which usually CHANGE the location in the real world.              |
|                        |                                                                                                                 |
|                        | Examples: vessels, trucks.                                                                                      |
|                        |                                                                                                                 |
|                        | *Note: dynamic resource nodes are not must for all scenarios.*                                                  |
+------------------------+-----------------------------------------------------------------------------------------------------------------+
 
Graph
-----

Graph module is a light wrapper for static resource nodes and dynamic resource nodes.
It use an matrix to organize the static and dynamic resource nodes, respectively.

.. code-block:: python

    graph = Graph(static_resource_node_num, dynamic_resource_node_num)
    # Instantiate a graph object according to node umber
    graph.setup()
    # Setup the graph
    graph.set_attribute(node_type, node_index, attribute_type, tick, value)
    # Set the value of an attribute on a specified node at specified tick
    value = graph.get_attribute(node_type, node_index, attribute_type, tick)
    # Get the value of an attribute on a specified node at specified tick

Snapshot List
-------------

Snapshot list is an abstraction of spatial-temporal graphs, which includes all spatial graph between two optimization events.
One snapshot is a past tick spatial graph backup and can be refreshed at each tick.

.. image:: ../images/snapshot.png

For outside user, snapshot lists can be fetched from the environment.
As a complete description of graph module,
the information of dynamic and static nodes can be obtained from corresponding parts of snapshot list, respectively.

.. code-block:: python

    snap_shot_list = env.snapshot_list
    # Current snapshot list can be fetched as a property of the environment.
    attribute_list = env.snapshot_list.attributes
    # Get all supported attributes list
    value = snapshot_list.static_nodes[[tick]: [node_id]: ([attribute_name], [attribute_slot])]
    # Get information about static nodes from snapshot list
    value = snapshot_list.dynamic_nodes[[tick]: [node_id]: ([attribute_name], [attribute_slot])]
    # Get information about dynamic nodes from snapshot list

