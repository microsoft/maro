Virtual Machine Scheduling (VM Scheduling)
===========================================

The Virtual Machine (VM) Scheduling scenario simulates the VM scheduling problem
in a data center. In a specific time window where a specific total number of VM
requests and arrival pattern is fixed, given a cluster of limited physical
machines(PM), different VM placement solutions result in different number of
successful completion and different operating cost for the data center.


Resource Flow
--------------

In this scenario, the physical resources in each physical machine (PM) is the
central resource, which currently includes the physical cores and memory. A full
resource life cycle always contains the steps below:

- Coming VM requests will ask for a specific amount of resource, different VM
  requests may have different requirement.
- According to the scheduling agent, the VM will be created in a specific PM if
  remaining resource enough.
- Later, the dynamic utilization change and the real-time energy consumption
  will be simulated in the runtime simulation of this VM.
- After a period of execution, the VM completes its tasks. The simulator will
  deallocate the resource allocated to this VM and delete this VM. Finally, the
  resource is free again and ready to serve the next VM request.

VM Request
^^^^^^^^^^^

In the VM scheduling scenario, the VM requests are uniformly sampled from real
workloads. As long as the original dataset is large enough and the sample ratio
not too small, the sampled VM requests can follow a similar distribution to the
original ones. We hope this can present you with the most real needs and provide
you with a demand simulation that is closest to the real situation.

VM Allocation
^^^^^^^^^^^^^^

Runtime Simulation
^^^^^^^^^^^^^^^^^^^

Dynamic Utilization
~~~~~~~~~~~~~~~~~~~~

Real-time Energy Consumption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VM Deallocation
^^^^^^^^^^^^^^^^


Topologies
-----------

Azure Topologies
^^^^^^^^^^^^^^^^^

Naive Baseline
^^^^^^^^^^^^^^^


Quick Start
------------

Data Preparation
^^^^^^^^^^^^^^^^^

Environment Interface
^^^^^^^^^^^^^^^^^^^^^^

DecisionEvent
~~~~~~~~~~~~~~

Action
~~~~~~~

Example
^^^^^^^^

Jump to `this notebook <>`_ for a quick experience.
