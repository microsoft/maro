
Grass Cluster Provisioning in On-Premises Environment
=====================================================

With the following guide, you can build up a MARO cluster in
`grass mode <../distributed_training/orchestration_with_grass.html#orchestration-with-grass>`_
in local private network and run your training job in On-Premises distributed environment.

Prerequisites
-------------

* Linux with Python 3.6+
* `Install Powershell <https://docs.microsoft.com/en-us/powershell/scripting/install/installing-powershell?view=powershell-7.1>`_ if you are using Windows Server

Cluster Management
------------------

* Create a cluster with a `deployment <#grass-cluster-create>`_

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

* Push your training image

  .. code-block:: sh

    # Push image 'my_image' to the cluster
    maro grass image push my_grass_cluster --image-name my_image

* Push your training data

  .. code-block:: sh

    # Push data under './my_training_data' to a relative path '/my_training_data' in the cluster
    # You can then assign your mapping location in the start-job deployment
    maro grass data push my_grass_cluster ./my_training_data/* /my_training_data

* Start a training job with a `deployment <#grass-start-job>`_

  .. code-block:: sh

    # Start a training job with a start-job deployment
    maro grass job start my_grass_cluster ./grass-start-job.yml

* Or, schedule batch jobs with a `deployment <#grass-start-schedule>`_

  .. code-block:: sh

    # Start a training schedule with a start-schedule deployment
    maro grass schedule start my_grass_cluster ./grass-start-schedule.yml

* Get the logs of the job

  .. code-block:: sh

    # Get the logs of the job
    maro grass job logs my_grass_cluster my_job_1

* List the current status of the job

  .. code-block:: sh

    # List the current status of the job
    maro grass job list my_grass_cluster

* Stop a training job

  .. code-block:: sh

    # Stop a training job
    maro grass job stop my_job_1

Sample Deployments
------------------

grass-cluster-create
^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   mode: grass/on-premises
   name: cluster_name

   user:
     admin_public_key: "{ssh public key with 'ssh-rsa' prefix}"
     admin_username: admin


grass-node-join
^^^^^^^^^^^^^^^

.. code-block:: yaml

    mode: "grass/on-premises"
    name: ""
    cluster: ""
    public_ip_address: ""
    hostname: ""
    system: "linux"
    resources:
      cpu: 1
      memory: 1024
      gpu: 0


grass-start-job
^^^^^^^^^^^^^^^

.. code-block:: yaml

   mode: grass
   name: my_job_1

   allocation:
     mode: single-metric-balanced
     metric: cpu

   components:
     actor:
       command: "bash {project root}/my_training_data/job_1/actor.sh"
       image: my_image
       mount:
         target: “{project root}”
       num: 5
       resources:
         cpu: 2
         gpu: 0
         memory: 2048m
     learner:
       command: "bash {project root}/my_training_data/job_1/learner.sh"
       image: my_image
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
   name: my_schedule_1

   allocation:
     mode: single-metric-balanced
     metric: cpu

   job_names:
     - my_job_2
     - my_job_3
     - my_job_4
     - my_job_5

   components:
     actor:
       command: "bash {project root}/my_training_data/job_1/actor.sh"
       image: my_image
       mount:
         target: “{project root}”
       num: 5
       resources:
         cpu: 2
         gpu: 0
         memory: 2048m
     learner:
       command: "bash {project root}/my_training_data/job_1/learner.sh"
       image: my_image
       mount:
         target: "{project root}"
       num: 1
       resources:
         cpu: 2
         gpu: 0
         memory: 2048m
