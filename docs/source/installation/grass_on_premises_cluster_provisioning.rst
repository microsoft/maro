.. _grass-on-premises-cluster-provisioning:

Grass Cluster Provisioning in On-Premises Environment
=====================================================

With the following guide, you can build up a MARO cluster in
:ref:`grass/on-premises <grass>`
in local private network and run your training job in On-Premises distributed environment.

Prerequisites
-------------

* Linux with Python 3.7+
* `Install Powershell <https://docs.microsoft.com/en-us/powershell/scripting/install/installing-powershell?view=powershell-7.1>`_ if you are using Windows Server
* For master node vm, need install flask, gunicorn, and redis.

Cluster Management
------------------

* Create a cluster with a :ref:`deployment <grass-on-premises-create>`

  .. code-block:: sh

    # Create a grass cluster with a grass-create deployment
    maro grass create ./grass-azure-create.yml

* Let a node join a specified cluster

  .. code-block:: sh

    # Let a worker node join into specified cluster
    maro grass node join ./node-join.yml

* Let a node leave a specified cluster

  .. code-block:: sh

    # Let a worker node leave a specified cluster
    maro grass node leave {cluster_name} {node_name}


* Delete the cluster

  .. code-block:: sh

    # Delete a grass cluster
    maro grass delete my_grass_cluster


Run Job
-------

See :ref:`Run Job in grass/azure <grass-azure-cluster-provisioning/run-job>` for reference.


Sample Deployments
------------------

grass-on-premises-create
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   mode: grass/on-premises
   name: clusterName

   user:
     admin_id: admin

   master:
     username: root
     hostname: maroMaster
     public_ip_address: 137.128.0.1
     private_ip_address: 10.0.0.4


grass-on-premises-join-cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    mode: grass/on-premises

    master:
      private_ip_address: 10.0.0.4

    node:
      hostname: maroNode1
      username: root
      public_ip_address: 137.128.0.2
      private_ip_address: 10.0.0.5
      resources:
        cpu: all
        memory: 2048m
        gpu: 0

     config:
       install_node_runtime: true
       install_node_gpu_support: false
