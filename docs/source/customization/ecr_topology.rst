ECR Topology
============

Currently, the predefined :doc:`ECR scenario <../scenario/ecr>` only
ready topology configurations under "maro/simulator/scenarios/ecr/topologies" (by folder name).


So if you need to add a customized topology, you should:


#. Create a folder named as you wish under "maro/simulator/scenarios/ecr/topologies", such as "dummy" for test.

#. Then add config.yml into "dummy" folder from step 1, and modify the configuration as you need (refer to following sample).

Now you have the new topology for ECR scenario, you can use it when initializing an environment.


.. code-block:: python

    from maro.simulator import env

    env = env("ecr", "dummy")


Dummy topology definition
`````````````````````

.. code-block:: yaml

    # int, total container number in this topology.
    total_containers: 100000
    # String value (fixed or unfixed).
    #    Fixed means the order number will be affected by containers that being used, the value will following the configured distribution.
    #    Unfixed means the order will consider how many containers are being used now, so the actual number depend on current container usage.
    order_generate_mode: fixed
    # list of 2 int number, 1st means how many past stops should be recorded in snapshot,
    # 2nd means how many future stops should be recorded.
    stop_number: [4, 3]
    # list of float, volume of each container size (only one container size for now).
    container_volumes: [1]

    # how many percentage of containers are occupied
    container_usage_proportion:
      # int number, period of total container usage proportion distribution function
      period: 112
      # list of tuple(tick, proportion) used for data generator to incorporate and generate orders at each tick
      sample_nodes:
      - [0, 0.008]
      - [111, 0.008]
      # noise to apply when generating orders
      sample_noise: 0

    # configuration about ports
    ports:
      # name of port
      port_001:
        # capacity of port
        capacity: 100000
        # value and noise of buffer ticks when full converted into empty
        empty_return:
          buffer_ticks: 1
          noise: 0
        # value and noise of buffer ticks when empty converted into full
        full_return:
          buffer_ticks: 1
          noise: 0
        # Proportion of this port taking total containers at the first tick
        initial_container_proportion: 0.25
        order_distribution:
          # Value and noise of proportion of orders generated ot this port
          source:
            noise: 0
            proportion: 0.33
          # Value and noise of proportion of orders for this port transporting to following ports
          targets:
            supply_port_001:
              noise: 0
              proportion: 1
      port_002:
        capacity: 1000000
        empty_return:
          buffer_ticks: 1
          noise: 0
        full_return:
          buffer_ticks: 1
          noise: 0
        initial_container_proportion: 0.25
        order_distribution:
          source:
            noise: 0
            proportion: 0

    # configuration of route
    routes:
      # name of route
      route_001:
      # stop configuration
      # distance from current port to next one
      - distance: 400
        # current port name
        port_name: port_001
      - distance: 400
        port_name: port_002

    # configuration of vessel
    vessels:
      # name of vessel
      vessel_001:
        # capacity of vessel
        capacity: 100000
        # ticks and noise that the vessel parking at a port
        parking:
          duration: 2
          noise: 0
        # which route this vessel belongs to
        route:
          # which port this vessel is parking at beginning
          initial_port_name: port_001
          route_name: route_001
        # sailing speed and noise configuration
        sailing:
          noise: 0
          speed: 50
