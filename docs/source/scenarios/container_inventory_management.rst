
Container Inventory Management (CIM)
====================================

The Container Inventory Management (CIM) scenario simulates a common problem of
container shipping in marine transportation. Imagine an international market:
The goods are packed in containers and shipped by vessels from the exporting
country to the importing country. As a result of the imbalanced global trade,
the volume of available containers in different ports may not match their needs.
In other words, some ports will have excess containers while some ports may be
in short. Therefore, We can use the excess capacity on vessels to reposition
empty containers to alleviate this imbalance.

Resource Flow
-------------

In this scenario, the **container** is the central resource. Two events will
trigger the movement of the container:

* The first one is the order, which will lead to the transportation of goods from
  the source port to the destination port.
* The second one is the repositioning operation. It is used to rebalance the
  container distribution worldwide.

.. image:: ../images/scenario/cim.container_flow.svg
   :target: ../images/scenario/cim.container_flow.svg
   :alt: The Life Cycle of the Container

Order
^^^^^

To simulate a real market, we assume that there will be a certain number of orders
from some ports to other ports every day. And the total order number of each day
is generated according to a predefined distribution. These orders are then allocated
to each export port in a relatively fixed proportion, and each export port will
have a relatively fixed number of import ports as customers. The order distribution
and the proportion of order allocation are specified in the topology and can be
customized based on different requirements.

An order will trigger a life cycle of a container, as shown in the figure above,
a life cycle is defined as follows:

* Once an order is generated, an empty container of the corresponding export port
  (source port) will be released to the shipper.
* The shipper will fill the container with cargo which turns it into a laden and
  then return it to the port after a few days.
* Loading occurs when the vessel arrives at this port.
* After several days of sailing, the vessel will finally arrive at the corresponding
  import port (destination port) where the discharging of the laden happens.
* Then the laden will be released to the consignee, and the consignee will take
  out the cargo in it, which turns it into an empty container again.
* Finally, the consignee returns it as an available container for the import port
  in a few days.

Container Repositioning
^^^^^^^^^^^^^^^^^^^^^^^

As mentioned above, to rebalance the container distribution, the agent in each
port will decide how to reposition the empty containers every time a vessel
arrives at the port. The decision consists of two parts:

* Whether to take a ``discharge`` operation or a ``load`` operation;
* The number of containers to discharge/load.

The export-oriented ports (e.g. the ports in China) show a clearly high demand
feature, and usually require additional supply of empty containers. These ports
will tend to discharge empty containers from the vessel if feasible. While the
import-oriented ports have a significant surplus feature, that usually receive
many empty container from the consignee. So the imported-oriented ports will tend
to load the surplus empty containers into the vessel if there is free capacity.

The specific quantity to operate for a ``discharge`` action is limited by the
remaining space in the port and the total number of empty containers in the vessel.
Similarly, a ``load`` action is limited by the remaining space in the vessel and
the total number of empty containers in the port. Of course, a good decision will
not only consider the self future supply and demand situation, but also the needs
and situation of the upstream and downstream ports.

Topologies
----------

To provide an exploration road map from easy to difficult, two kinds of topologies
are designed and provided in CIM scenario. Toy topologies provide simplified
environments for algorithm debugging and will show some typical relationships
between ports to users. We hope these will provide users with some insights to
know more and deeper about this scenario. While the global topologies are based
on the real-world data, and are bigger and more complicated to present the real
problem.

Toy Topologies
^^^^^^^^^^^^^^

.. image:: ../images/scenario/cim.toys.svg
   :target: ../images/scenario/cim.toys.svg
   :alt: CIM toy topologies

*(In these topologies, the solid lines indicate the service route (voyage) among
ports, while the dashed lines indicate the container flow triggered by orders.)*

**toy.4p_ssdd_l0.D**\ : There are four ports in this topology. According to the orders,
D1 and D2 are simple demanders (the port requiring additional empty containers)
while S2 is a simple supplier (the port with surplus empty containers). Although
S1 is a simple destination port, it's at the intersection of two service routes,
which makes it a special port in this topology. To achieve the global optimum,
S1 must learn to distinguish the service routes and take service route specific
repositioning operations.

**toy.5p_ssddd_l0.D**\ : There are five ports in this topology. According to the orders,
D1 and D2 are simple demanders while S1 and S2 are simple suppliers. As a port
in the intersection of service routes, although the supply and demand of port T1
can reach a state of self-balancing, it still plays an important role for the
global optimum. The best repositioning policy for port T1 is to transfer the
surplus empty containers from the left service route to the right one. Also, the
port D1 and D2 should learn to discharge only the quantity they need and leave the
surplus ones to other ports.

**toy.6p_sssbdd_l0.D**\ : There are six ports in this topology. Similar to toy.5p_ssddd,
in this topology, there are simple demanders D1 and D2, simple suppliers S1 and
S2, and self-balancing ports T1 and T2. More difficult than in toy.5p_ssddd,
more transfers should be taken to reposition the surplus empty containers from
the left most service route to the right most one, which means a multi-steps
solution that involving more ports is required.

Global Topologies
^^^^^^^^^^^^^^^^^

**global_trade.22p_l0.D**\ : This is a topology designed based on the real-world data.
The order generation model in this topology is built based on the trade data from
`WTO <https://data.wto.org/>`_. According to the query results in WTO from January
2019 to October 2019, The ports with large trade volume are selected, and the
proportion of each port as the source of orders is directly proportional to the
export volume of the country it belongs to, while the proportion as the destination
is proportional to the import volume.
In this scenario, there are much more ports, much more service routes. And most
ports no longer have a simple supply/demand feature. The cooperation among ports
is much more complex and it is difficult to find an efficient repositioning policy
manually.

.. image:: ../images/scenario/cim.global_trade.svg
   :target: ../images/scenario/cim.global_trade.svg
   :alt: global_trade.22p

*(To make it clearer, the figure above only shows the service routes among ports.)*

Naive Baseline
^^^^^^^^^^^^^^

Below are the final environment metrics of the method *no repositioning* and
*random repositioning* in different topologies. For each experiment, we setup
the environment and test for a duration of 1120 ticks (days).

No Repositioning
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Topology
     - Total Requirement
     - Resource Shortage
     - Repositioning Number
   * - toy.4p_ssdd_l0.0
     - 2,240,000
     - 2,190,000
     - 0
   * - toy.4p_ssdd_l0.1
     - 2,240,000
     - 2,190,000
     - 0
   * - toy.4p_ssdd_l0.2
     - 2,240,000
     - 2,190,000
     - 0
   * - toy.4p_ssdd_l0.3
     - 2,239,460
     - 2,189,460
     - 0
   * - toy.4p_ssdd_l0.4
     - 2,244,068
     - 2,194,068
     - 0
   * - toy.4p_ssdd_l0.5
     - 2,244,068
     - 2,194,068
     - 0
   * - toy.4p_ssdd_l0.6
     - 2,244,068
     - 2,194,068
     - 0
   * - toy.4p_ssdd_l0.7
     - 2,244,068
     - 2,194,068
     - 0
   * - toy.4p_ssdd_l0.8
     - 2,241,716
     - 2,191,716
     - 0

|

.. list-table::
   :header-rows: 1

   * - Topology
     - Total Requirement
     - Resource Shortage
     - Repositioning Number
   * - toy.5p_ssddd_l0.0
     - 2,240,000
     - 2,140,000
     - 0
   * - toy.5p_ssddd_l0.1
     - 2,240,000
     - 2,140,000
     - 0
   * - toy.5p_ssddd_l0.2
     - 2,240,000
     - 2,140,000
     - 0
   * - toy.5p_ssddd_l0.3
     - 2,239,460
     - 2,139,460
     - 0
   * - toy.5p_ssddd_l0.4
     - 2,244,068
     - 2,144,068
     - 0
   * - toy.5p_ssddd_l0.5
     - 2,244,068
     - 2,144,068
     - 0
   * - toy.5p_ssddd_l0.6
     - 2,244,068
     - 2,144,068
     - 0
   * - toy.5p_ssddd_l0.7
     - 2,244,068
     - 2,144,068
     - 0
   * - toy.5p_ssddd_l0.8
     - 2,241,716
     - 2,141,716
     - 0

|

.. list-table::
   :header-rows: 1

   * - Topology
     - Total Requirement
     - Resource Shortage
     - Repositioning Number
   * - toy.6p_sssbdd_l0.0
     - 2,240,000
     - 2,087,000
     - 0
   * - toy.6p_sssbdd_l0.1
     - 2,240,000
     - 2,087,000
     - 0
   * - toy.6p_sssbdd_l0.2
     - 2,240,000
     - 2,087,000
     - 0
   * - toy.6p_sssbdd_l0.3
     - 2,239,460
     - 2,086,460
     - 0
   * - toy.6p_sssbdd_l0.4
     - 2,244,068
     - 2,091,068
     - 0
   * - toy.6p_sssbdd_l0.5
     - 2,244,068
     - 2,091,068
     - 0
   * - toy.6p_sssbdd_l0.6
     - 2,244,068
     - 2,091,068
     - 0
   * - toy.6p_sssbdd_l0.7
     - 2,244,068
     - 2,091,068
     - 0
   * - toy.6p_sssbdd_l0.8
     - 2,241,716
     - 2,088,716
     - 0

|

.. list-table::
   :header-rows: 1

   * - Topology
     - Total Requirement
     - Resource Shortage
     - Repositioning Number
   * - global_trade.22p_l0.0
     - 2,240,000
     - 1,028,481
     - 0
   * - global_trade.22p_l0.1
     - 2,240,000
     - 1,081,935
     - 0
   * - global_trade.22p_l0.2
     - 2,240,000
     - 1,083,358
     - 0
   * - global_trade.22p_l0.3
     - 2,239,460
     - 1,085,212
     - 0
   * - global_trade.22p_l0.4
     - 2,244,068
     - 1,089,628
     - 0
   * - global_trade.22p_l0.5
     - 2,244,068
     - 1,102,913
     - 0
   * - global_trade.22p_l0.6
     - 2,244,068
     - 1,122,092
     - 0
   * - global_trade.22p_l0.7
     - 2,244,068
     - 1,162,108
     - 0
   * - global_trade.22p_l0.8
     - 2,241,716
     - 1,161,714
     - 0

Random Repositioning
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Topology
     - Total Requirement
     - Resource Shortage
     - Repositioning Number
   * - toy.4p_ssdd_l0.0
     - 2,240,000
     - 1,497,138 :math:`\pm`  30,423
     - 4,185,080 :math:`\pm` 185,140
   * - toy.4p_ssdd_l0.1
     - 2,240,000
     - 1,623,710 :math:`\pm`  36,421
     - 2,018,360 :math:`\pm`  36,700
   * - toy.4p_ssdd_l0.2
     - 2,240,000
     - 1,501,466 :math:`\pm`  48,566
     - 2,145,180 :math:`\pm`  90,300
   * - toy.4p_ssdd_l0.3
     - 2,239,460
     - 1,577,011 :math:`\pm`  35,109
     - 2,098,500 :math:`\pm`  35,120
   * - toy.4p_ssdd_l0.4
     - 2,244,068
     - 1,501,835 :math:`\pm` 103,196
     - 2,180,480 :math:`\pm`  33,020
   * - toy.4p_ssdd_l0.5
     - 2,244,068
     - 1,546,227 :math:`\pm`  81,107
     - 2,077,320 :math:`\pm` 113,740
   * - toy.4p_ssdd_l0.6
     - 2,244,068
     - 1,578,863 :math:`\pm` 127,815
     - 2,220,720 :math:`\pm` 106,660
   * - toy.4p_ssdd_l0.7
     - 2,244,068
     - 1,519,495 :math:`\pm`  60,555
     - 2,441,480 :math:`\pm`  79,700
   * - toy.4p_ssdd_l0.8
     - 2,241,716
     - 1,603,063 :math:`\pm` 109,149
     - 2,518,920 :math:`\pm` 193,200

|

.. list-table::
   :header-rows: 1

   * - Topology
     - Total Requirement
     - Resource Shortage
     - Repositioning Number
   * - toy.5p_ssddd_l0.0
     - 2,240,000
     - 1,371,021 :math:`\pm`  34,619
     - 3,966,120 :math:`\pm` 138,960
   * - toy.5p_ssddd_l0.1
     - 2,240,000
     - 1,720,068 :math:`\pm`  18,939
     - 1,550,280 :math:`\pm`  25,600
   * - toy.5p_ssddd_l0.2
     - 2,240,000
     - 1,716,435 :math:`\pm`  15,499
     - 1,496,860 :math:`\pm`  31,260
   * - toy.5p_ssddd_l0.3
     - 2,239,460
     - 1,700,456 :math:`\pm`  26,510
     - 1,586,640 :math:`\pm`  11,500
   * - toy.5p_ssddd_l0.4
     - 2,244,068
     - 1,663,139 :math:`\pm`  34,244
     - 1,594,160 :math:`\pm` 103,040
   * - toy.5p_ssddd_l0.5
     - 2,244,068
     - 1,681,519 :math:`\pm` 107,863
     - 1,635,360 :math:`\pm`  61,880
   * - toy.5p_ssddd_l0.6
     - 2,244,068
     - 1,660,330 :math:`\pm`  38,318
     - 1,630,060 :math:`\pm`  81,580
   * - toy.5p_ssddd_l0.7
     - 2,244,068
     - 1,709,022 :math:`\pm`  31,440
     - 1,854,340 :math:`\pm` 167,080
   * - toy.5p_ssddd_l0.8
     - 2,241,716
     - 1,763,950 :math:`\pm`  73,935
     - 1,858,420 :math:`\pm`  60,680

|

.. list-table::
   :header-rows: 1

   * - Topology
     - Total Requirement
     - Resource Shortage
     - Repositioning Number
   * - toy.6p_sssbdd_l0.0
     - 2,240,000
     - 1,529,774 :math:`\pm` 73,104
     - 3,989,560 :math:`\pm` 232,740
   * - toy.6p_sssbdd_l0.1
     - 2,240,000
     - 1,736,385 :math:`\pm` 16,736
     - 1,122,120 :math:`\pm`  28,960
   * - toy.6p_sssbdd_l0.2
     - 2,240,000
     - 1,765,945 :math:`\pm`  4,680
     - 1,052,520 :math:`\pm`  44,020
   * - toy.6p_sssbdd_l0.3
     - 2,239,460
     - 1,811,987 :math:`\pm` 15,436
     - 998,740 :math:`\pm`  69,680
   * - toy.6p_sssbdd_l0.4
     - 2,244,068
     - 1,783,362 :math:`\pm` 39,122
     - 1,059,860 :math:`\pm`  49,100
   * - toy.6p_sssbdd_l0.5
     - 2,244,068
     - 1,755,551 :math:`\pm` 44,855
     - 1,101,100 :math:`\pm`  55,180
   * - toy.6p_sssbdd_l0.6
     - 2,244,068
     - 1,830,504 :math:`\pm` 10,690
     - 1,141,660 :math:`\pm`  10,520
   * - toy.6p_sssbdd_l0.7
     - 2,244,068
     - 1,742,129 :math:`\pm` 23,910
     - 1,311,420 :math:`\pm`  64,560
   * - toy.6p_sssbdd_l0.8
     - 2,241,716
     - 1,761,283 :math:`\pm` 22,338
     - 1,336,540 :math:`\pm`  30,020

|

.. list-table::
   :header-rows: 1

   * - Topology
     - Total Requirement
     - Resource Shortage
     - Repositioning Number
   * - global_trade.22p_l0.0
     - 2,240,000
     - 1,010,009 :math:`\pm` 20,942
     - 548,240 :math:`\pm`  14,600
   * - global_trade.22p_l0.1
     - 2,240,000
     - 1,027,395 :math:`\pm` 19,183
     - 188,160 :math:`\pm`  12,940
   * - global_trade.22p_l0.2
     - 2,240,000
     - 1,035,851 :math:`\pm`  4,352
     - 181,240 :math:`\pm`   5,240
   * - global_trade.22p_l0.3
     - 2,239,460
     - 1,032,480 :math:`\pm`  1,332
     - 190,220 :math:`\pm`   8,920
   * - global_trade.22p_l0.4
     - 2,244,068
     - 1,034,412 :math:`\pm` 11,689
     - 186,080 :math:`\pm`   6,280
   * - global_trade.22p_l0.5
     - 2,244,068
     - 1,042,869 :math:`\pm` 16,146
     - 188,720 :math:`\pm`   7,880
   * - global_trade.22p_l0.6
     - 2,244,068
     - 1,096,502 :math:`\pm` 26,896
     - 302,280 :math:`\pm`  27,540
   * - global_trade.22p_l0.7
     - 2,244,068
     - 1,144,981 :math:`\pm`  5,355
     - 283,520 :math:`\pm`  25,700
   * - global_trade.22p_l0.8
     - 2,241,716
     - 1,154,184 :math:`\pm`  7,043
     - 270,960 :math:`\pm`   2,240

Quick Start
-----------

Data Preparation
^^^^^^^^^^^^^^^^

To start a simulation in CIM scenario, no extra data processing is needed. You
can just specify the scenario and the topology when initialize an environment and
enjoy your exploration in this scenario.

Environment Interface
^^^^^^^^^^^^^^^^^^^^^

Before starting interaction with the environment, we need to know the definition
of ``DecisionEvent`` and ``Action`` in CIM scenario first. Besides, you can query the
environment `snapshot list <../key_components/data_model.html#advanced-features>`_
to get more detailed information for the decision making.

DecisionEvent
~~~~~~~~~~~~~

Once the environment need the agent's response to promote the simulation, it will
throw an ``DecisionEvent``. In the scenario of CIM, the information of each
``DecisionEvent`` is listed as below:

* **tick** (int): The corresponding tick.
* **port_idx** (int): The id of the port/agent that needs to respond to the
  environment.
* **vessel_idx** (int): The id of the vessel/operation object of the port/agent.
* **action_scope** (ActionScope): ActionScope has two attributes:

  * ``load`` indicates the maximum quantity that can be loaded from the port the
    vessel.
  * ``discharge`` indicates the maximum quantity that can be discharged from the
    vessel to the port.

* **early_discharge** (int): When the available capacity in the vessel is not
  enough to load the ladens, some of the empty containers in the vessel will be
  early discharged to free the space. The quantity of empty containers that have
  been early discharged due to the laden loading is recorded in this field.

Action
~~~~~~

Once we get a ``DecisionEvent`` from the environment, we should respond with an
``Action``. Valid ``Action`` could be:

* ``None``\ , which means do nothing.
* A valid ``Action`` instance, including:

  * **vessel_idx** (int): The id of the vessel/operation object of the port/agent.
  * **port_idx** (int): The id of the port/agent that take this action.
  * **action_type** (ActionType): Whether to load or discharge empty containers in this action.
  * **quantity** (int): The (non-negative) quantity of empty containers to be loaded/discharged.

Example
^^^^^^^

Here we will show you a simple example of interaction with the environment in
random mode, we hope this could help you learn how to use the environment interfaces:

.. code-block:: python
  from maro.simulator import Env
  from maro.simulator.scenarios.cim.common import Action, ActionType, DecisionEvent

  from random import randint

  # Initialize an Env for cim scenario
  env = Env(scenario="cim", topology="toy.5p_ssddd_l0.0", start_tick=0, durations=100)

  metrics: object = None
  decision_event: DecisionEvent = None
  is_done: bool = False
  action: Action = None

  # Start the env with a None Action
  metrics, decision_event, is_done = env.step(None)

  while not is_done:
      # Generate a random Action according to the action_scope in DecisionEvent
      action_scope = decision_event.action_scope
      to_discharge = action_scope.discharge > 0 and randint(0, 1) > 0

      action = Action(
          decision_event.vessel_idx,
          decision_event.port_idx,
          randint(0, action_scope.discharge if to_discharge else action_scope.load),
          ActionType.DISCHARGE if to_discharge else ActionType.LOAD
      )

      # Respond the environment with the generated Action
      metrics, decision_event, is_done = env.step(action)

Jump to `this notebook <https://github.com/microsoft/maro/tree/master/notebooks/container_inventory_management/interact_with_environment.ipynb>`_
for a quick experience.

Visualization
-------------

The resource holders in this scenario is the port and vessel.
In order to facilitate users to select specific data and
observe the overall or partial data trend, the
visualization tool provides data selection options in two dimensions:
Inter-epoch view & Intra-epoch view.

Inter-epoch view
^^^^^^^^^^^^^^^^

To change "Start Epoch" and "End Epoch", user could specify the selected
data range. To change "Epoch Sampling Ratio", user could change the
sampling rate of selected data.

.. image:: ../images/visualization/dashboard/cim_inter_epoch.gif
   :alt: cim_inter_epoch

Intra-epoch view
^^^^^^^^^^^^^^^^

This part shows the data under a selected epoch. By scrolling the
slider, users can select different epochs. Furthermore, this part of
data is divided into two dimensions: by snapshot and by port according
to time and space. In terms of data display, according to the different
types of attributes, it is divided into two levels: accumulated data
(accumulated attributes. e.g. acc\_fulfillment) and detail data.

If user choose to view information by ports, attributes of the selected
port would be displayed.

.. image:: ../images/visualization/dashboard/cim_intra_epoch_by_ports.gif
   :alt: cim_intra_epoch_by_ports

If user choose to view data by snapshots, attributes of selected
snapshot would be displayed. The charts and data involved in this part
are relatively rich, and we will introduce them by level.

Accumulated Data
~~~~~~~~~~~~~~~~

This part includes the transfer volume heat map, bar chart of port
accumulated attributes and top-k ports of different attributes.

As shown in the following example, the x-axis and y-axis of transfer
volume heat map refers to terminal port index and start port index
respectively. The rect refers to the volume of cargoes transfer from
start port to terminal port. By changing the snapshot index, user could
view the dynamic changes in the volume of cargo delivered by the port
over time in the current epoch.

The bar chart of Port Accumulated Attributes displays the global change of
ports.

.. image:: ../images/visualization/dashboard/cim_intra_epoch_by_snapshot_acc_data.gif
   :alt: cim_intra_epoch_by_snapshot_acc_data

Detail Data
~~~~~~~~~~~

Since the cargoes is transported through vessels, information of vessels
could be viewed by snapshot. Same as ports, user could change the sampling rate of vessels.

.. image:: ../images/visualization/dashboard/cim_intra_epoch_by_snapshot_detail_data.gif
   :alt: cim_intra_epoch_by_snapshot_detail_data
