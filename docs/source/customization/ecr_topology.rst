ECR Topology
=============

Currently, the predefined :doc:`ECR scenario <../scenario/ecr>` only
ready topology configurations under "maro/simulator/scenarios/ecr/topologies" (by folder name).


So if you need to add a customized topology, you should:


#. Create a folder named as you wish under "maro/simulator/scenarios/ecr/topologies", such as "dummy" for test.

#. Then copy config.yml from "maro/simulator/scenarios/ecr/4p_ssdd_l0.0" into "dummy" folder from step 1, and modify the configuration as you need.

Now you have the new topology for ECR scenario, you can use it when initializing an environment.


.. code-block:: python

    from maro.simulator import env

    env = env("ecr", "dummy")




.. ecr_configs_desc:

ECR configuration fields
------------------------

total_containers
````````````````
    int, total container number in this topology.


container_volumes
`````````````````
    List of float, volume of each container size (only one container size for now).


stop_number
```````````
    List of 2 int number, 1st means how many past stops should be recorded in snapshot,
    2nd means how many future stops should be recorded.


order_generate_mode
```````````````````
    String value (fixed or unfixed).
    Fixed means the order number will be affected by containers that being used, the value will following the configured distribution.
    Unfixed means the order will consider how many containers are being used now, so the actual number depend on current container usage.


container_usage_proportion
``````````````````````````
    How many percentage of containers are occupied

period
''''''
    Int number, period of total container usage proportion distribution function

sample_nodes
''''''''''''
    List of tuple(tick, proportion) used for data generator to incorporate and generate orders at each tick

sample_noise
''''''''''''
    Noise to apply when generating orders


ports
`````
    Configurations about ports

vessels
```````
    Configurations about vessels

routes
``````
    Configurations about routes
