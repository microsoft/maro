Virtual Machine Scheduling (VM Scheduling)
===========================================

The Virtual Machine (VM) Scheduling scenario simulates the VM scheduling problem
in a cloud data center. Within a specific time, the number of VM 
requests and arrival pattern is fixed. Given a cluster of limited physical
machines(PM), different VM placement strategeies result in different amount of
successful completion and different operating cost for the data center. For cloud proivders, a 
good VM allocation strategy can maximize the resourece utilization and thus can increase the profit by 
providing more VMs to users. For cloud users, a good VM allocation strategy can 
minimize the VM response time and have a better using experience. We hope this scenario can meet 
the real needs and provide you with a demand simulation that is closest to the real situation.


Resource Flow
--------------

In this scenario, the physical resources in each physical machine (PM) are the
central resource, which currently includes the physical cores and memory. A full
resource life cycle always contains the steps below:

- Coming VM requests ask for a certain amount of resources. Resource requirements are varied
  based on the different VM requests.
- Based on the scheduling agent's strategy, the VM will be allocated to and be created
  in a specified PM as long as that PM's remaining resources are enough.
- The VM's resource utilization changes dynamically and the PM's real-time energy consumption
  will be simulated in the runtime simulation.
- After a period of execution, the VM completes its tasks. The simulator will release the resources
  allocated to this VM and deallocate this VM from the PM. 
  Finally, the resource is free and is ready to serve the next VM request.

VM Request
^^^^^^^^^^^

In the VM scheduling scenario, the VM requests are uniformly sampled from real
workloads. As long as the original dataset is large enough and the sample ratio
is not too small, the sampled VM requests can follow a similar distribution to the
original ones. 

Given a fixed time interval, a VM request will arise according to the real VM workload data and it  
might also be the request sent by the ``PostponeAction``. The request contains the VM information and 
the buffer time. 

* Whenever receive a VM request, the MARO simulator will first calculate the 
  remaining resources of each PM. The required resources includes CPU cores and memory. 
* Then, the simulator will generate a ``PendingDecision`` event with the ``DecisionPayload``, which 
  contains all valid PMs (valid here means that the remaining resources are enough), and the 
  information of the awaiting VM.

VM Allocation
^^^^^^^^^^^^^^

The agent will make the decision among all valid PMs after get the ``DecisionPayload``.
The agent will decide one PM to host the VM based on the given strategy. Afterwards, the agent 
will send the ``Action`` back to the simulator for the following simulation. 
There are three different valid ``Action`` in current VM Scheduling scenario. 

* **None**: If the MARO simulator receives the **None** Action, it will do nothing and ignore the VM request.
* ``AllocateAction``: If the MARO simulator receives the ``AllocateAction``, the VM's creation time will be 
  fixed at that tick. Besides, the simulator will update the workloads (the workloads include CPU cores,
  the memory, and the energy consumption) of the target PM.
* ``PostponeAction``: If the MARO simulator receives the ``PostponeAction``, it will calculate the 
  remaining buffer time. 

  * If the time is still enough, the simulator will re-generate a new requirement
    event and insert it to the corresponding tick (based on the ``Postpone Step`` and ``DELAY_DURATION``). 
    The ``DecisionPayload`` of the new requirement event only differs in the remaining buffer time from the 
    old ones.
  * If the time is exhausted, the simulator will note it as a failed allocation.

See the detailed attributes `here <#action>`_.

Runtime Simulation
^^^^^^^^^^^^^^^^^^^

Dynamic Utilization
~~~~~~~~~~~~~~~~~~~~

To make the simulated environment closest to the real situation. We also simulate CPU utilization of each
VM. The CPU utilization of the VM varies every tick based on the real VM workload readings. 
We will regularly calculate the total resources (CPU utilization) of each PM in every tick and update 
to the PM workload for the following decision.

Real-time Energy Consumption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the most important characteristics that cloud providers concern is the PM's enery consumption. As we
mention before, the lower energy consumption of the PMs, the lower cost to maintain the physical servers. In 
our simulation, we currently simulate the energy based on the CPU utilization. In short, PM cost more energy
if it has higher CPU utilization. [`Reference <https://dl.acm.org/doi/10.1145/1273440.1250665>`_]

VM Deallocation
^^^^^^^^^^^^^^^^

The MARO simulator regularly checks the finished VMs in every tick. It will then release the finished VM's
resources, including CPU cores and memory, and finally remove the VM from the PM.

Topologies
-----------

To provide samples from easy to difficult, two kinds of simple topologies are designed and 
provided in VM Scheduling scenario. 

Azure Topologies
^^^^^^^^^^^^^^^^^

The original data are provided by `Azure public dataset 
<https://github.com/Azure/AzurePublicDataset>`_. In our scenario, we pre-processed the AzurePublicDatasetV2. 
The dataset contains real Azure VM workloads, including the information of VMs and their utilization readings 
in 2019 lasting for 30 days. The original dataset contains 2,695,548 VMs.

The detailed information of the data schema can be found
`here <https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md>`_. After pre-processed,
we only retain real VM creation and deletion time (converted to the tick, 1 tick means 5 minutes in real time),
VM cores and memory(GB) requirements, and we also renumber the original VM ID.
As for the utilization readings part, we store the renumbered VM ID and VM's CPU utilization sorting by the timestamp (tick).

**azure.2019.10k**\ : We randomly sampled 10,000 VMs from the AzurePublicDatasetV2.

**azure.2019.336k**\ : We randomly sampled 336,000 VMs from the AzurePublicDatasetV2.

Naive Baseline
^^^^^^^^^^^^^^^

Belows are the final environment metrics of the method random allocation and best-fit allocation in 
different topologies. For each experiment, we setup the environment and test for a duration of 30 days.


Random Allocation
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Topology
     - PM Setting
     - Total VM Requests
     - Total Energy Consumption
     - Successful Allocation
     - Successful completion
     - Failed Allocation
   * - Azure.2019.10k 
     - 100 PMs, 32 Cores, 128 GB
     - 10,000
     - 2,430,651.6
     - 9,850
     - 9,030
     - 150
   * - 
     - 100 PMs, 16 Cores, 112 GB
     - 10,000
     - 2,978,445.0
     - 8,011
     - 7,411
     - 1,989
   * - Azure.2019.336k
     - 880 PMs, 32 Cores, 128 GB
     - 335,985
     - 26,681,249.7
     - 176,468
     - 165,715
     - 159,517
   * -  
     - 880 PMs, 16 Cores, 112 GB
     - 335,985
     - 26,367,238.7
     - 92,885
     - 87,153
     - 243,100

Best-Fit Allocation
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Topology
     - PM Setting
     - Total VM Requests
     - Total Energy Consumption
     - Successful Allocation
     - Successful completion
     - Failed Allocation
   * - Azure.2019.10k 
     - 100 PMs, 32 Cores, 128 GB
     - 10,000
     - 2,395,328.7
     - 10,000
     - 9,180
     - 0
   * - 
     - 100 PMs, 16 Cores, 112 GB
     - 10,000
     - 2,987,086.6
     - 7,917
     - 7,313
     - 2,083
   * - Azure.2019.336k 
     - 880 PMs, 32 Cores, 128 GB
     - 335,985
     - 26,695,470.8
     - 171,044
     - 160,495
     - 164,941
   * - 
     - 880 PMs, 16 Cores, 112 GB
     - 335,985
     - 26,390,972.9
     - 92,263
     - 86,600
     - 243,722



Quick Start
------------

Data Preparation
^^^^^^^^^^^^^^^^^

When the environment is first created, the system will automatically trigger the pipeline to download 
and process the data files. Afterwards, if you want to run multiple simulations, the system will detect
whether the processed data files exist or not. If not, it will then trigger the pipeline again. Otherwise,
the system will reuse the processed data files. 


Environment Interface
^^^^^^^^^^^^^^^^^^^^^^

Before starting interaction with the environment, we need to know the definition of ``DecisionPayload`` and 
``Action`` in VM Scheduling scenario first. Besides, you can query the environment snapshot list to get more 
detailed information for the decision making.

DecisionPayload
~~~~~~~~~~~~~~

Once the environment need the agent's response to promote the simulation, it will throw an ``PendingDecision``
event with the ``DecisionPayload``. In the scenario of VM Scheduling, the information of ``DecisionPayload`` is 
listed as below:

* **valid_pms** (List[int]): The list of the PM ID that is considered as valid (Its CPU and memory resource is enough for the incoming VM request).
* **vm_id** (int): The VM ID of the incoming VM request (VM request that is waiting for the allocation).
* **vm_cpu_cores_requirement** (int): The CPU cores that is requested by the incoming VM request.
* **vm_memory_requirement** (int): The memory resource that is reqeusted by the incoming VM request.
* **remaining_buffer_time** (int): The remaining buffer time for the VM allocation. The VM request will be treated as failed when the remaining_buffer_time is spent. The initial buffer time budget can be set in the config.yml.

Action
~~~~~~~

Once get a ``PendingDecision`` event from the envirionment, the agent should respond with an Action. Valid 
``Action`` includes:

* **None**. It means do nothing but ignore this VM request.
* ``AllocateAction``. It includes:

  * vm_id (int): The ID of the VM that is waiting for the allocation.
  * pm_id (int): The ID of the PM where the VM is scheduled to allocate to.
* ``PostponeAction``. It includes:

  * vm_id (int): The ID of the VM that is waiting for the allocation.
  * postpone_step (int): The number of times that the allocation to be postponed. The unit 
    is ``DELAY_DURATION``. 1 means delay 1 ``DELAY_DURATION``, which can be set in the config.yml.

Example
^^^^^^^^

Here we will show you a simple example of interaction with the environment in random mode, we 
hope this could help you learn how to use the environment interfaces:

.. code-block:: python

  import random

  from maro.simulator import Env
  from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

  # Initialize an Env for vm_scheduling scenario
  env = Env(
    scenario="vm_scheduling",
    topology="azure.2019.10k",
    start_tick=0,
    durations=8638,
    snapshot_resolution=1
  )

  metrics: object = None
  decision_event: DecisionPayload = None
  is_done: bool = False
  action: AllocateAction = None
      
  # Start the env with a None Action
  metrics, decision_event, is_done = env.step(None)

  while not is_done:
      valid_pm_num: int = len(decision_event.valid_pms)
      if valid_pm_num <= 0:
          # No valid PM now, postpone.
          action: PostponeAction = PostponeAction(
              vm_id=decision_event.vm_id,
              postpone_step=1
          )
      else:
          # Randomly choose an available PM.
          random_idx = random.randint(0, valid_pm_num - 1)
          pm_id = decision_event.valid_pms[random_idx]
          action: AllocateAction = AllocateAction(
              vm_id=decision_event.vm_id,
              pm_id=pm_id
          )
      metrics, decision_event, is_done = env.step(action)

  print(f"[Random] Topology: azure.2019.10k. Total ticks: 8638. Start tick: 0")
  print(metrics)

Jump to `this notebook <>`_ for a quick experience.
