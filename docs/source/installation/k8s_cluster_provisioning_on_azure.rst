
K8S Cluster Provisioning on Azure
=================================

With the following guide, you can build up a MARO cluster in
`k8s mode <../distributed_training/orchestration_with_k8s.html#orchestration-with-k8s>`_
on Azure and run your training job in a distributed environment.

Prerequisites
-------------


* `Install the Azure CLI and login <https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest>`_
* `Install and set up kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl/>`_
* `Install docker <https://docs.docker.com/engine/install/>`_ and
  `Configure docker to make sure it can be managed as a non-root user <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_

Cluster Management
------------------


* Create a cluster with a `deployment <#k8s-azure-create>`_

.. code-block:: sh

   # Create a k8s cluster
   maro k8s create ./k8s-azure-create.yml


* Scale the cluster

.. code-block:: sh

   # Scale nodes with 'Standard_D4s_v3' specification to 2
   maro k8s node scale my_k8s_cluster Standard_D4s_v3 2

Check `VM Size <https://docs.microsoft.com/en-us/azure/virtual-machines/sizes>`_
to see more node specifications.


* Delete the cluster

.. code-block:: sh

   # Delete a k8s cluster
   maro k8s delete my_k8s_cluster

Run Job
-------


* Push your training image

.. code-block:: sh

   # Push image 'my_image' to the cluster
   maro k8s image push my_k8s_cluster --image-name my_image


* Push your training data

.. code-block:: sh

   # Push data under './my_training_data' to a relative path '/my_training_data' in the cluster
   # You can then assign your mapping location in the start-job deployment
   maro k8s data push my_k8s_cluster ./my_training_data/* /my_training_data


* Start a training job with a `deployment <#k8s-start-job>`_

.. code-block:: sh

   # Start a training job with a start-job deployment
   maro k8s job start my_k8s_cluster ./k8s-start-job.yml


* Or, schedule batch jobs with a `deployment <#k8s-start-schedule>`_

.. code-block:: sh

   # Start a training schedule with a start-schedule deployment
   maro k8s schedule start my_k8s123_cluster ./k8s-start-schedule.yml


* Get the logs of the job

.. code-block:: sh

   # Logs will be exported to current directory
   maro k8s job logs my_k8s_cluster my_job_1


* List the current status of the job

.. code-block:: sh

   # List current status of jobs
   maro k8s job list my_k8s_cluster my_job_1


* Stop a training job

.. code-block:: sh

   # Stop a training job
   maro k8s job stop my_k8s_cluster my_job_1

Sample Deployments
------------------

k8s-azure-create
^^^^^^^^^^^^^^^^

.. code-block:: yaml

   mode: k8s
   name: my_k8s_cluster

   cloud:
     infra: azure
     location: eastus
     resource_group: my_k8s_resource_group
     subscription: my_subscription

   user:
     admin_public_key: "{ssh public key with 'ssh-rsa' prefix}"
     admin_username: admin

   master:
     node_size: Standard_D2s_v3

k8s-start-job
^^^^^^^^^^^^^

.. code-block:: yaml

   mode: k8s
   name: my_job_1

   components:
     actor:
       command: ["bash", "{project root}/my_training_data/actor.sh"]
       image: my_image
       mount:
         target: "{project root}"
       num: 5
       resources:
         cpu: 2
         gpu: 0
         memory: 2048m
     learner:
       command: ["bash", "{project root}/my_training_data/learner.sh"]
       image: my_image
       mount:
         target: "{project root}"
       num: 1
       resources:
         cpu: 2
         gpu: 0
         memory: 2048m

k8s-start-schedule
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   mode: k8s
   name: my_schedule_1

   job_names:
     - my_job_2
     - my_job_3
     - my_job_4
     - my_job_5

   components:
     actor:
       command: ["bash", "{project root}/my_training_data/actor.sh"]
       image: my_image
       mount:
         target: "{project root}"
       num: 5
       resources:
         cpu: 2
         gpu: 0
         memory: 2048m
     learner:
       command: ["bash", "{project root}/my_training_data/learner.sh"]
       image: my_image
       mount:
         target: "{project root}"
       num: 1
       resources:
         cpu: 2
         gpu: 0
         memory: 2048m
