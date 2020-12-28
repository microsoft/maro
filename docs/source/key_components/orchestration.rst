
Distributed Orchestration
=========================

MARO provides easy-to-use CLI commands to provision and manage training clusters
on cloud computing service like `Azure <https://azure.microsoft.com/en-us/>`_.
These CLI commands can also be used to schedule the training jobs with the
specified resource requirements. In MARO, all training job related components
are dockerized for easy deployment and resource allocation. It provides a unified
abstraction/interface for different orchestration framework
(e.g. `Grass <#grass>`_\ , `Kubernetes <#kubernetes>`_\ ).

.. image:: ../images/distributed/orch_overview.svg
   :target: ../images/distributed/orch_overview.svg
   :alt: Orchestration Overview
   :width: 600

Process
-------
The process mode is designed for starting the training job by multi-processes
and simulating the real distributed cluster. Depending on the Redis, The process
mode realizes job management and self-control. The reason for designing the process
mode is to reduce the difficulty of running jobs in the distributed cluster.
It has the following advantages:

* No need deployment.
* Easy to use.
* Lightweight, no other dependencies are required.

In the Process mode:

* All jobs will be started by multi-processes and managed by MARO Process CLI.
* Customized settings support, such as Redis, the number of parallel running jobs,
  and agents check interval.
* For each job's start/stop, a ticket will be pushed into the job queue in Redis.
  The agents will check those job queues periodically, and start/stop jobs depending
  on the tickets in the job queues.

.. image:: ../images/distributed/orch_process.svg
   :target: ../images/distributed/orch_process.svg
   :alt: Orchestration Overview
   :width: 600

Grass
-----

Grass is a self-designed, development purpose orchestration framework. It can be
confidently applied to small/middle size cluster (< 200 nodes). The design goal
of Grass is to speed up the distributed algorithm prototype development.
It has the following advantages:

* Fast deployment in a small cluster.
* Fine-grained resource management.
* Lightweight, no other dependencies are required.

In the Grass mode:

* All VMs will be deployed in the same virtual network for a faster, more stable
  connection and larger bandwidth. Please note that the maximum number of VMs is
  limited by the `available dedicated IP addresses <https://docs.microsoft.com/en-us/azure/virtual-network/virtual-networks-faq#what-address-ranges-can-i-use-in-my-vnets>`_.
* It is a centralized topology, the master node will host Redis service for peer
  discovering, Fluentd service for log collecting, SMB service for file sharing.
* On each VM, the probe (worker) agent is used to track the computing resources
  and detect abnormal events.

Check `Grass Cluster Provisioning on Azure <../installation/grass_cluster_provisioning_on_azure.html>`_
to get how to use it.

.. image:: ../images/distributed/orch_grass.svg
   :target: ../images/distributed/orch_grass.svg
   :alt: Orchestration Grass Mode in Azure
   :width: 600

Kubernetes
----------

MARO also supports Kubernetes (k8s) as an orchestration option.
With this widely used framework, you can easily build up your training cluster
with hundreds and thousands of nodes. It has the following advantages:

* Higher durability.
* Better scalability.

In the Kubernetes mode:

* The dockerized job component runs in Kubernetes pod, and each pod only hosts
  one component.
* All Kubernetes pods are registered into the same virtual network using
  `Container Network Interface(CNI) <https://github.com/containernetworking/cni>`_.

Check `K8S Cluster Provisioning on Azure <../installation/k8s_cluster_provisioning_on_azure.html>`_
to get how to use it.

.. image:: ../images/distributed/orch_k8s.svg
   :target: ../images/distributed/orch_k8s.svg
   :alt: Orchestration K8S Mode in Azure
   :width: 600
