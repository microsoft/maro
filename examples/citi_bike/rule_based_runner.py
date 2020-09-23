import argparse
import csv
import heapq
import os
import random
import numpy as np
from maro.simulator import Env
from maro.simulator.scenarios.citi_bike.common import Action, DecisionType


DIR = "../../maro/simulator/scenarios/citi_bike/topologies"
DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MINUTES_PER_DAY = 1440


def get_nyc_topologies(year: str, month: str):
    if year == "all":
        return [tp for tp in os.listdir(DIR) if tp.startswith("ny")]
    if month == "all":
        topologies = [tp for tp in os.listdir(DIR) if tp.startswith(f"ny.{year}")]
        if not topologies:
            print(f"No data available for year {year}")
        return topologies

    if 0 < int(month) < 10:
        month = "0" + month
    return [f"ny.{year}{month}"]


def get_toy_topologies():
    return [tp for tp in os.listdir(DIR) if tp.startswith("toy")]


class NullAgent:
    def __init__(self):
        pass

    def choose_action(self, decision_event):
        return None


class RandomAgent:
    def __init__(self):
        pass

    def choose_action(self, decision_event):
        other = random.choice([station_id for station_id in decision_event.action_scope.keys()
                               if station_id != decision_event.station_idx])
        num = random.randrange(decision_event.action_scope[other] + 1)
        if decision_event.type == DecisionType.Supply:
            return Action(from_station_idx=decision_event.station_idx, to_station_idx=other, number=num)
        else:
            return Action(from_station_idx=other, to_station_idx=decision_event.station_idx, number=num)


class GreedyAgent:
    def __init__(self, top_k: int = 1):
        self._top_k = top_k

    def choose_action(self, decision_event):
        if decision_event.type == DecisionType.Supply:
            # find k target stations with the most empty slots, randomly choose one of them and send as many bikes to
            # it as allowed by the action scope
            top_k_demands = []
            for demand_candidate, available_docks in decision_event.action_scope.items():
                if demand_candidate == decision_event.station_idx:
                    continue
                heapq.heappush(top_k_demands, (available_docks, demand_candidate))
                if len(top_k_demands) > self._top_k:
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
                if len(top_k_supplies) > self._top_k:
                    heapq.heappop(top_k_supplies)
            max_reposition, source_idx = random.choice(top_k_supplies)
            action = Action(source_idx, decision_event.station_idx, max_reposition)

        return action


def main(args):
    # get topologies and output file name
    if args.topology == 'nyc':
        topologies = get_nyc_topologies(args.year, args.month)
        file_name = "-".join(['nyc', args.year, args.month, str(args.duration), args.mode])
    elif args.topology == 'toy':
        topologies = get_toy_topologies()
        file_name = "-".join(['toy', str(args.duration), args.mode])
    else:
        raise ValueError(f"Unsupported topology: {args.topology}")

    # add appropriate headers to output file
    header = ["topology"] + ["requirement", "shortage", "cost"] * (len(args.seed) if args.mode == "random" else 1)
    if not os.path.exists(file_name):
        with open(file_name, 'a') as fp:
            writer = csv.writer(fp)
            writer.writerow(header)

    # create an agent of specified mode
    if args.mode == "null":
        agent = NullAgent()
    elif args.mode == "random":
        agent = RandomAgent()
    elif args.mode == "greedy":
        agent = GreedyAgent(args.k)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    def run(env, agent, seed=None):
        env.reset()
        if seed is not None:
            env.set_seed(seed)
            np.random.seed(seed)
        _, decision_event, done = env.step(None)
        while not done:
            metrics, decision_event, done = env.step(agent.choose_action(decision_event))

        return env.metrics

    print(f"Topologies: {topologies} (total {len(topologies)})")
    print(f"duration: {args.duration}")
    print(f"action mode: {args.mode}")
    for tp in topologies:
        days = args.duration if args.duration >= 0 else DAYS[int(tp[-2:])-1]
        print(f"Running topology {tp} for {days} days")
        env = Env(scenario="citi_bike", topology=tp, start_tick=0, durations=days * MINUTES_PER_DAY,
                  snapshot_resolution=10)
        results = [tp]
        if args.mode == "null" or (args.mode == "greedy" and args.k == 1):  # no use for seed
            result = run(env, agent)
            results.extend([result["trip_requirements"], result["bike_shortage"], result["operation_cost"]])
        else:
            for i, seed in enumerate(args.seed):
                result = run(env, agent, seed)
                results.extend([result["trip_requirements"], result["bike_shortage"], result["operation_cost"]])

        with open(file_name, 'a') as fp:
            writer = csv.writer(fp)
            writer.writerow(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("topology", help="supported modes: nyc, toy")
    parser.add_argument("mode", help="supported modes: random, null, greedy")
    parser.add_argument("-y", "--year", default="all", help="year")
    parser.add_argument("-m", "--month", default="all", help="month")
    parser.add_argument("-s", "--seed", type=int, default=[0], action="append", help="random seed")
    parser.add_argument("-d", "--duration", type=int, default=-1, help="duration")
    parser.add_argument("-k", type=int, default=1, help="top k supply or demand candidates")
    args = parser.parse_args()
    main(args)
