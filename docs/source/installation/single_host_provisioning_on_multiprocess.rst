Single Host Provisioning on Multi-process
=========================================
With the following guide, it is easy to implement your training jobs through
multi-process in the single-host environment.

Prerequisites
-------------
* Liunx with Python 3.6+
* Redis

Single-host Management
----------------------
* Create a single-host environment with `setting-deployment <#process-setting-deployment>`_

  .. code-block:: sh

    # Create single-host multi-process environment
    maro process create ./process_setting_deployment.yml

* Delete a single-host environment

  .. code-block:: sh

    # Delete single-host multi-process environment
    maro process delete

* Get deployment template with setting deployment to ./target/path

  .. code-block:: sh

    # Get deployment template
    maro process template --setting_deploy ./target/path


* Start a training job with `job-deployment <#process-job-deployment>`_

  .. code-block:: sh

    # Start a training job
    maro process job start ./process_job_deployment.yml

* Stop a training job

  .. code-block:: sh

    # Stop a training job with job_name
    maro process job stop job_name

* Delete a job

  .. code-block:: sh

    # Delete a job
    maro process job delete job_name

* List all jobs

  .. code-block:: sh

    # List all jobs
    maro process job list

* Get job's log with job_name, dumps to cwd

  .. code-block:: sh

    # Get job's log
    maro process job logs job_name

* Start a schedule with `schedule-deployment <#process-schedule-deployment>`_

  .. code-block:: sh

    # Start a schedule
    maro process schedule start ./process_schedule_deployment.yml

* Stop a schedule

  .. code-block:: sh

    # Stop a schedule with schedule name
    maro process schedule stop schedule_name

Sample Deployments
------------------

process-setting-deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    redis_info:
        host: "localhost"
        port: 19999
    parallel_level: 1
    keep_agent_alive: 1
    check_interval: 60
    redis_mode: MARO
    agent_countdown: 5

process-job-deployment
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    mode: process
    name: MyJobName

    components:
        actor:
            num: 5
            command: "python /mnt/data/run_actor.py"
        learner:
            num: 1
            command: "python /mnt/data/run_learner.py"

process-schedule-deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    mode: process
    name: MyScheduleName

    job_names:
        - MyJobName2
        - MyJobName3
        - MyJobName4
        - MyJobName5

    components:
        actor:
            num: 5
            command: "python /mnt/data/run_actor.py"
        learner:
            num: 1
            command: "python /mnt/data/run_learner.py"
