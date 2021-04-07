.. _grass-azure-cluster-provisioning:

Grass Cluster Provisioning on Azure
===================================

With the following guide, you can build up a MARO cluster in
:ref:`grass/azure <grass>`
mode on Azure and run your training job in a distributed environment.

Prerequisites
-------------

* `Install the Azure CLI (preferred version: v2.20.0) and login <https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest>`_
* `Install docker <https://docs.docker.com/engine/install/>`_ and
  `Configure docker to make sure it can be managed as a non-root user <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_

Cluster Management
------------------

* Create a cluster with a :ref:`deployment <#grass-azure-create>`

  .. code-block:: sh

    # Create a grass cluster with a grass-create deployment
    maro grass create ./grass-azure-create.yml

* Scale the cluster

    Check `VM Size <https://docs.microsoft.com/en-us/azure/virtual-machines/sizes>`_ to see more node specifications.

  .. code-block:: sh

    # Scale nodes with 'Standard_D4s_v3' specification to 2
    maro grass node scale myGrassCluster Standard_D4s_v3 2

    # Scale nodes with 'Standard_D2s_v3' specification to 0
    maro grass node scale myGrassCluster Standard_D2s_v3 0

* Delete the cluster

  .. code-block:: sh

    # Delete a grass cluster
    maro grass delete myGrassCluster

* Start/Stop nodes to save costs

  .. code-block:: sh

    # Start 2 nodes with 'Standard_D4s_v3' specification
    maro grass node start myGrassCluster Standard_D4s_v3 2

    # Stop 2 nodes with 'Standard_D4s_v3' specification
    maro grass node stop myGrassCluster Standard_D4s_v3 2

* Get statuses of the cluster

  .. code-block:: sh

    # Get master status
    maro grass status myGrassCluster master

    # Get nodes status
    maro grass status myGrassCluster nodes

    # Get containers status
    maro grass status myGrassCluster containers

* Clean up the cluster

    Delete all running jobs, schedules, containers in the cluster.

  .. code-block:: sh

    maro grass clean myGrassCluster

.. _grass-azure-cluster-provisioning/run-job:

Run Job
-------

* Push your training image from local machine

  .. code-block:: sh

    # Push image 'myImage' to the cluster,
    # 'myImage' is a docker image that loaded on the machine that executed this command
    maro grass image push myGrassCluster --image-name myImage

* Push your training data

  .. code-block:: sh

    # Push dqn folder under './myTrainingData/' to a relative path '/myTrainingData' in the cluster
    # You can then assign your mapping location in the start-job-deployment
    maro grass data push myGrassCluster ./myTrainingData/dqn /myTrainingData

* Start a training job with a :ref:`start-job-deployment <grass-start-job>`

  .. code-block:: sh

    # Start a training job with a start-job deployment
    maro grass job start myGrassCluster ./grass-start-job.yml

* Or, schedule batch jobs with a :ref:`start-schedule-deployment <grass-start-schedule>`

    These jobs will shared the same specification of components.

    A best practice to use this command will be:
    Push your training configs all at once with "``maro grass data push``",
    and get the jobName from environment variables in the containers,
    then use the specific training config based on the jobName.

  .. code-block:: sh

    # Start a training schedule with a start-schedule deployment
    maro grass schedule start myGrassCluster ./grass-start-schedule.yml

* Get the logs of the job

  .. code-block:: sh

    # Get the logs of the job
    maro grass job logs myGrassCluster myJob1

* List the current status of the job

  .. code-block:: sh

    # List the current status of the job
    maro grass job list myGrassCluster

* Stop a training job

  .. code-block:: sh

    # Stop a training job
    maro grass job stop myJob1

Sample Deployments
------------------

grass-azure-create
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   mode: grass/azure
   name: myGrassCluster

   cloud:
     resource_group: myResourceGroup
     subscription: mySubscription
     location: eastus
     default_username: admin
     default_public_key: "{ssh public key}"

   user:
     admin_id: admin

   master:
     node_size: Standard_D2s_v3

grass-start-job
^^^^^^^^^^^^^^^

    You can replace {project root} with a valid linux path. e.g. /home/admin

    Then the data you push will be mount into this folder.

.. code-block:: yaml

   mode: grass
   name: myJob1

   allocation:
     mode: single-metric-balanced
     metric: cpu

   components:
     actor:
       command: "python {project root}/myTrainingData/dqn/job1/start_actor.py"
       image: myImage
       mount:
         target: "{project root}"
       num: 5
       resources:
         cpu: 1
         gpu: 0
         memory: 1024m
     learner:
       command: "python {project root}/myTrainingData/dqn/job1/start_learner.py"
       image: myImage
       mount:
         target: "{project root}"
       num: 1
       resources:
         cpu: 2
         gpu: 0
         memory: 2048m

grass-start-schedule
^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   mode: grass
   name: mySchedule1

   allocation:
     mode: single-metric-balanced
     metric: cpu

   job_names:
     - myJob2
     - myJob3
     - myJob4
     - myJob5

   components:
     actor:
       command: "python {project root}/myTrainingData/dqn/schedule1/actor.py"
       image: myImage
       mount:
         target: “{project root}”
       num: 5
       resources:
         cpu: 1
         gpu: 0
         memory: 1024m
     learner:
       command: "bash {project root}/myTrainingData/dqn/schedule1/learner.py"
       image: myImage
       mount:
         target: "{project root}"
       num: 1
       resources:
         cpu: 2
         gpu: 0
         memory: 2048m
