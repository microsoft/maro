.. _k8s-aks-cluster-provisioning:

K8S Cluster Provisioning on Azure
=================================

With the following guide, you can build up a MARO cluster in
:ref:`k8s/aks <k8s>`
on Azure and run your training job in a distributed environment.

Prerequisites
-------------

* `Install the Azure CLI and login <https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest>`_
* `Install and set up kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl/>`_
* `Install docker <https://docs.docker.com/engine/install/>`_ and
  `Configure docker to make sure it can be managed as a non-root user
  <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_
* `Download AzCopy <https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10>`_,
  then move the AzCopy executable to /bin folder or
  add the directory location of the AzCopy executable to your system path:

.. code-block:: sh

    # Take AzCopy version 10.6.0 as an example

    # Linux
    tar xvf ./azcopy_linux_amd64_10.6.0.tar.gz; cp ./azcopy_linux_amd64_10.6.0/azcopy /usr/local/bin

    # MacOS (may required MacOS Security & Privacy setting)
    unzip ./azcopy_darwin_amd64_10.6.0.zip; cp ./azcopy_darwin_amd64_10.6.0/azcopy /usr/local/bin

    # Windows
    # 1. Unzip ./azcopy_windows_amd64_10.6.0.zip
    # 2. Add the path of ./azcopy_windows_amd64_10.6.0 folder to your Environment Variables
    # Ref: https://superuser.com/questions/949560/how-do-i-set-system-environment-variables-in-windows-10


Cluster Management
------------------

* Create a cluster with a :ref:`deployment <k8s-aks-create>`

  .. code-block:: sh

    # Create a k8s cluster
    maro k8s create ./k8s-azure-create.yml

* Scale the cluster

  .. code-block:: sh

      Check `VM Size <https://docs.microsoft.com/en-us/azure/virtual-machines/sizes>`_ to see more node specifications.

    # Scale nodes with 'Standard_D4s_v3' specification to 2
    maro k8s node scale myK8sCluster Standard_D4s_v3 2

    # Scale nodes with 'Standard_D2s_v3' specification to 0
    maro k8s node scale myK8sCluster Standard_D2s_v3 0

* Delete the cluster

  .. code-block:: sh

    # Delete a k8s cluster
    maro k8s delete myK8sCluster

Run Job
-------

* Push your training image

  .. code-block:: sh

    # Push image 'myImage' to the cluster
    maro k8s image push myK8sCluster --image-name myImage

* Push your training data

  .. code-block:: sh

    # Push dqn folder under './myTrainingData/' to a relative path '/myTrainingData' in the cluster
    # You can then assign your mapping location in the start-job-deployment
    maro k8s data push myGrassCluster ./myTrainingData/dqn /myTrainingData

* Start a training job with a :ref:`deployment <k8s-start-job>`

  .. code-block:: sh

    # Start a training job with a start-job-deployment
    maro k8s job start myK8sCluster ./k8s-start-job.yml

* Or, schedule batch jobs with a :ref:`deployment <k8s-start-schedule>`

  .. code-block:: sh

    # Start a training schedule with a start-schedule-deployment
    maro k8s schedule start myK8sCluster ./k8s-start-schedule.yml

* Get the logs of the job

  .. code-block:: sh

    # Logs will be exported to current directory
    maro k8s job logs myK8sCluster myJob1

* List the current status of the job

  .. code-block:: sh

    # List current status of jobs
    maro k8s job list myK8sCluster myJob1

* Stop a training job

  .. code-block:: sh

    # Stop a training job
    maro k8s job stop myK8sCluster myJob1

Sample Deployments
------------------

k8s-aks-create
^^^^^^^^^^^^^^

.. code-block:: yaml

   mode: k8s/aks
   name: myK8sCluster

   cloud:
     subscription: mySubscription
     resource_group: myResourceGroup
     location: eastus
     default_public_key: "{ssh public key}"
     default_username: admin

   master:
     node_size: Standard_D2s_v3

k8s-start-job
^^^^^^^^^^^^^

.. code-block:: yaml

   mode: k8s/aks
   name: myJob1

   components:
     actor:
       command: ["python", "{project root}/myTrainingData/dqn/start_actor.py"]
       image: myImage
       mount:
         target: "{project root}"
       num: 5
       resources:
         cpu: 2
         gpu: 0
         memory: 2048M
     learner:
       command: ["python", "{project root}/myTrainingData/dqn/start_learner.py"]
       image: myImage
       mount:
         target: "{project root}"
       num: 1
       resources:
         cpu: 2
         gpu: 0
         memory: 2048M

k8s-start-schedule
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   mode: k8s/aks
   name: mySchedule1

   job_names:
     - myJob2
     - myJob3
     - myJob4
     - myJob5

   components:
     actor:
       command: ["python", "{project root}/myTrainingData/dqn/start_actor.py"]
       image: myImage
       mount:
         target: "{project root}"
       num: 5
       resources:
         cpu: 2
         gpu: 0
         memory: 2048M
     learner:
       command: ["python", "{project root}/myTrainingData/dqn/start_learner.py"]
       image: myImage
       mount:
         target: "{project root}"
       num: 1
       resources:
         cpu: 2
         gpu: 0
         memory: 2048M
