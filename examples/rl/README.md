# Reinforcement Learning (RL) Examples

This folder contains scenarios that employ reinforcement learning. MARO's RL toolkit makes it possible to use a common workflow on different scenarios, so long as the necessary scenario-related components are provided. The workflow consists of Python scripts for running the necessary components in single-threaded and distributed modes under ``workflows``. General scenario-independent settings can be found in ``config.yml``. The scenario can be chosen by setting the ``scenario`` field in this file.

## How to Run

Scripts to run the common workflow in docker containers are in ``scripts/docker``. Start by choosing "single", "sync" or "async" for the ``mode`` field in ``config.yml`` to run a scenario in single-threaded, synchronous and asynchronous modes, respectively. Go to this folder and execute ``bash run.sh`` to launch the program and Docker Compose will take care of starting the necessary containers. Note that the script will build the docker image first if it has not already been built by running ``bash build.sh``. When the program is finished, be sure to run ``bash kill.sh`` to clean up the containers and remove the network.

## Create Your Own Scenarios

The workflow scripts make it easy to create your own scenarios by only supplying the necessary ingredients without worrying about putting them together. The ingredients include:
* Definitions of state, action and reward shaping logic pertinent to your simulator and policies.
These definitions should be encapsulated in ``get_env_sampler``, which is a function that takes no parameters and returns an environment sampler;
* Definitions of policies and agent-to-policy mappings. These definitions should be provided as a dictionary named ``policy_func_index`` that maps policy names to functions that create them. A policy creating function takes a name and return a policy instance with that name;

Optionally, if you want to 
