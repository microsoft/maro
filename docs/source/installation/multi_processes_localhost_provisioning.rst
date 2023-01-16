Multi-processes Localhost Provisioning
=========================================
With the following guide, it is easy to implement your training jobs through
the multi-processes in the localhost environment.

Prerequisites
-------------
* Linux with Python 3.7+
* Redis

Cluster Management
----------------------
* Get job/schedule deployment template to ./target/path.

  .. code-block:: sh

    # Get deployment template
    maro process template ./target/path

* Create a localhost environment by default parameters.

  .. code-block:: sh

    # Create localhost multi-process environment
    maro process create

* Create a localhost environment with `setting-deployment <#process-setting-deployment>`_.

  .. code-block:: sh

    # Get process_setting_deployment to current path.
    maro process template --setting_deploy .

    maro process create ./process_setting_deployment.yml

* Delete a localhost environment.

  .. code-block:: sh

    # Delete localhost multi-process environment
    maro process delete

* Start a training job with `job-deployment <#process-job-deployment>`_.

  .. code-block:: sh

    # Start a training job
    maro process job start ./process_job_deployment.yml

* Stop a training job.

  .. code-block:: sh

    # Stop a training job with job_name
    maro process job stop job_name

* Delete a job including remove job details in Redis.

  .. code-block:: sh

    # Delete a job
    maro process job delete job_name

* List all jobs.

  .. code-block:: sh

    # List all jobs
    maro process job list

* Get job's log with job_name, dumps to current path.

  .. code-block:: sh

    # Get job's log
    maro process job logs job_name

* Start a schedule with `schedule-deployment <#process-schedule-deployment>`_.

  .. code-block:: sh

    # Start a schedule
    maro process schedule start ./process_schedule_deployment.yml

* Stop a schedule.

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
    redis_mode: MARO      # one of [MARO, customized]. customized Redis won't be exited after maro process clear.
    parallel_level: 1     # Represented the maximum number of running jobs in the same times.
    keep_agent_alive: 1   # If 1 represented the agents won't exit until the environment delete; otherwise, 0.
    agent_countdown: 5    # After agent_countdown times checks, still no jobs will close agents. Available only if keep_agent_alive is 0.
    check_interval: 60    # The time interval (seconds) of agents check with Redis

process-job-deployment
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    mode: process
    name: MyJobName

    components:
        actor:
            num: 5
            command: "python /target/path/run_actor.py"
        learner:
            num: 1
            command: "python /target/path/run_learner.py"

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
            command: "python /target/path/run_actor.py"
        learner:
            num: 1
            command: "python /target/path/run_learner.py"
