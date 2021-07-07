# Reinforcement Learning (RL) Examples

This folder contains scenarios that employ reinforcement learning. It is possible to apply a common workflow to different scenarios, thanks to the abstractions from MARO's RL toolkit. The workflow can be found under ``workflows``, which contains Python scripts for running the necessary components in single-threaded and distributed modes. The scenario can be chosen by setting the ``scenario`` field in ``config.yml``.

## How to Run

To run the single-threaded workflow, go to ``workflows`` and run ``python3 simple_learner.py``. Alternatively, go to ``scripts`` and run ``bash run.sh``.

To run the distributed workflow, start by choosing "sync" or "async" for the ``mode`` field in ``config.yml``. The ``scrpts/docker`` folder provides bash scripts for simulating the distributed workflow using multiple docker containers on a single host. Go to this folder and execute ``bash run.sh`` to launch the program and Docker Compose will take care of starting the necessary containers. Note that the script will build the docker image first if it has not already been built by running ``bash build.sh``. When the program is finished, be sure to run ``bash kill.sh`` to clean up the containers and remove the network.

## Write Your Own Scenarios

To use the workflow provided under ``workflows``, the following ingredients are required:
* ``get_env_wrapper``, a function that takes no parameters and returns an environment wrapper instance.
* ``get_agent_wrapper``, a function that takes no parameters and returns an agent wrapper instance.
* ``policy_func_index``, a dictionary mapping policy names to functions that create them.
The policy-creating functions should take as its sole parameter a flag indicating whether the created policy is for roll-out or training. 
We recommend that you follow the ``cim`` example to write your own scenario.   
