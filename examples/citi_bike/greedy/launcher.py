import heapq
import io
import random

import yaml

from maro.simulator import Env
from maro.simulator.scenarios.citi_bike.common import Action, DecisionEvent, DecisionType
from maro.utils import convert_dottable

with io.open("config.yml", "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)


class GreedyPolicy:
    def __init__(self, supply_top_k: int = 1, demand_top_k: int = 1):
        """
        Agent that executes a greedy policy. If the event type is supply, send as many bikes as possible to one of the
        demand_k stations with the most empty slots. If the event type is demand, request as many bikes as possible from
        one of the supply_k stations with the most bikes.

        Args:
            supply_top_k (int): number of top supply candidates to choose from.
            demand_top_k (int): number of top demand candidates to choose from.
        """
        self._supply_top_k = supply_top_k
        self._demand_top_k = demand_top_k

    def choose_action(self, decision_event: DecisionEvent):
        if decision_event.type == DecisionType.Supply:
            # find k target stations with the most empty slots, randomly choose one of them and send as many bikes to
            # it as allowed by the action scope
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
            # find k source stations with the most bikes, randomly choose one of them and request as many bikes from
            # it as allowed by the action scope
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


if __name__ == "__main__":
    env = Env(scenario=config.env.scenario, topology=config.env.topology, start_tick=config.env.start_tick,
              durations=config.env.durations, snapshot_resolution=config.env.resolution)

    if config.env.seed is not None:
        env.set_seed(config.env.seed)

    policy = GreedyPolicy(config.agent.supply_top_k, config.agent.demand_top_k)
    metrics, decision_event, done = env.step(None)
    while not done:
        metrics, decision_event, done = env.step(policy.choose_action(decision_event))

    print(f"Greedy agent policy performance: {env.metrics}")

    env.reset()
