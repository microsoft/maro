Greedy Policy for Citi Bike
================================================

In this example we demonstrate using a simple greedy policy for `Citi Bike <https://maro.readthedocs.io/en/latest/scenarios/citi_bike.html>`_,
which is a real-world bike repositioning scenario.

Greedy Policy
-------------

Our greedy policy is simple: if the event type is supply, the policy will make
the current station send as many bikes as possible to one of k stations with the most empty docks. If the event type is
demand, the policy will make the current station request as many bikes as possible from one of k stations with the most
bikes. We use a heap data structure to find the top k supply/demand candidates from the action scope associated with
each decision event.

.. code-block:: python

    class GreedyPolicy:
        ...
        def choose_action(self, decision_event: DecisionEvent):
            if decision_event.type == DecisionType.Supply:
                """
                Find k target stations with the most empty slots, randomly choose one of them and send as many bikes to
                it as allowed by the action scope
                """
                top_k_demands = []
                for demand_candidate, available_docks in decision_event.action_scope.items():
                    if demand_candidate == decision_event.station_idx:
                        continue

                    heapq.heappush(top_k_demands, (available_docks, demand_candidate))
                    if len(top_k_demands) > self._demand_top_k:
                        heapq.heappop(top_k_demands)

                max_reposition, target_station_idx = random.choice(top_k_demands)
                action = Action(decision_event.station_idx, target_station_idx, max_reposition)
            else:
                """
                Find k source stations with the most bikes, randomly choose one of them and request as many bikes from
                it as allowed by the action scope.
                """
                top_k_supplies = []
                for supply_candidate, available_bikes in decision_event.action_scope.items():
                    if supply_candidate == decision_event.station_idx:
                        continue

                    heapq.heappush(top_k_supplies, (available_bikes, supply_candidate))
                    if len(top_k_supplies) > self._supply_top_k:
                        heapq.heappop(top_k_supplies)

                max_reposition, source_idx = random.choice(top_k_supplies)
                action = Action(source_idx, decision_event.station_idx, max_reposition)

            return action


Interaction with the Greedy Policy
----------------------------------

This environment is driven by `real trip history data <https://s3.amazonaws.com/tripdata/index.html>`_ from Citi Bike.

.. code-block:: python

    env = Env(scenario=config.env.scenario, topology=config.env.topology, start_tick=config.env.start_tick,
              durations=config.env.durations, snapshot_resolution=config.env.resolution)

    if config.env.seed is not None:
        env.set_seed(config.env.seed)

    policy = GreedyPolicy(config.agent.supply_top_k, config.agent.demand_top_k)
    metrics, decision_event, done = env.step(None)
    while not done:
        metrics, decision_event, done = env.step(policy.choose_action(decision_event))

    env.reset()

.. note::

  All related code snippets are supported in `maro playground <https://hub.docker.com/r/maro2020/playground>`_.
