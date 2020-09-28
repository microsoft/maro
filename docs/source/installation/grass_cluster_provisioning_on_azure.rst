
Grass Cluster Provisioning on Azure
===================================

With the following guide, you can build up a MARO cluster in
`grass mode <../distributed_training/orchestration_with_grass.html#orchestration-with-grass>`_
on Azure and run your training job in a distributed environment.

Prerequisites
-------------


* `Install the Azure CLI and login <https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest>`_
* `Install docker <https://docs.docker.com/engine/install/>`_ and
  `Configure docker to make sure it can be managed as a non-root user <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_

Cluster Management
------------------


* Create a cluster with a `deployment <#grass-azure-create>`_

.. code-block:: sh

   # Create a grass cluster with a grass-create deployment
   maro grass create ./grass-azure-create.yml


* Scale the cluster

.. code-block:: sh

   # Scale nodes with 'Standard_D4s_v3' specification to 2
   maro grass node scale my_grass_cluster Standard_D4s_v3 2

Check `VM Size <https://docs.microsoft.com/en-us/azure/virtual-machines/sizes>`_
to see more node specifications.


* Delete the cluster

.. code-block:: sh

   # Delete a grass cluster
   maro grass delete my_grass_cluster


* Start/stop nodes to save costs

.. code-block:: sh

   # Start 2 nodes with 'Standard_D4s_v3' specification
   maro grass node start my_grass_cluster Standard_D4s_v3 2

   # Stop 2 nodes with 'Standard_D4s_v3' specification
   maro grass node stop my_grass_cluster Standard_D4s_v3 2

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

grass-azure-create
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   mode: grass
   name: my_grass_cluster

   cloud:
     infra: azure
     location: eastus
     resource_group: my_grass_resource_group
     subscription: my_subscription

   user:
     admin_public_key: "{ssh public key with 'ssh-rsa' prefix}"
     admin_username: admin

   master:
     node_size: Standard_D2s_v3

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
