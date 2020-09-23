import argparse
import csv
import os
import random
import numpy as np
from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action, DecisionEvent


DIR = "../../maro/simulator/scenarios/cim/topologies"
TOPOLOGY_TERMINOLOGY = {'4': "4p_ssdd", '5': "5p_ssddd", '6': "6p_sssbdd"}


def get_toy_topologies(num_ports: str, level: str):
    if num_ports == "all":
        return [tp for tp in os.listdir(DIR) if tp.startswith("toy")]
    if level == "all":
        topologies = [tp for tp in os.listdir(DIR) if tp.startswith(f"toy.{TOPOLOGY_TERMINOLOGY[num_ports]}")]
        if not topologies:
            print(f"No topology available for {num_ports} ports")
        return topologies

    return [f"toy.{TOPOLOGY_TERMINOLOGY[num_ports]}_l0.{level}"]


def get_global_topologies(level: str):
    if level == "all":
        topologies = [tp for tp in os.listdir(DIR) if tp.startswith("global")]
        if not topologies:
            print(f"No global trade topology available")
        return topologies

    return [f"global_trade.22p_l0.{level}"]


class NullAgent:
    def __init__(self):
        pass

    def choose_action(self, decision_event, snapshot_list):
        return None


class RandomAgent:
    def __init__(self):
        pass

    def choose_action(self, decision_event: DecisionEvent, snapshot_list):
        scope = decision_event.action_scope
        tick = decision_event.tick
        port_idx = decision_event.port_idx
        vessel_idx = decision_event.vessel_idx

        action = random.randint(-10, 10) / 10
        port_empty = snapshot_list["ports"][tick: port_idx: ["empty", "full", "on_shipper", "on_consignee"]][0]
        vessel_remaining_space = snapshot_list["vessels"][tick: vessel_idx: ["empty", "full", "remaining_space"]][2]
        early_discharge = snapshot_list["vessels"][tick:vessel_idx: "early_discharge"][0]
        if action < 0:  # load
            actual_action = max(round(action * port_empty), -vessel_remaining_space)
        elif action > 0:  # discharge
            plan_action = action * (scope.discharge + early_discharge) - early_discharge
            actual_action = round(plan_action) if plan_action > 0 else round(action*scope.discharge)
        else:
            actual_action = 0

        return Action(vessel_idx, port_idx, actual_action)


def main(args):
    # get topologies and output file name
    if args.topology == "toy":
        topologies = get_toy_topologies(args.ports, args.level)
        file_name = "-".join(['ecr', 'toy', str(args.ports), str(args.level), args.mode])
    elif args.topology == "global":
        topologies = get_global_topologies(args.level)
        file_name = "-".join(['ecr', 'global', str(args.ports), str(args.level), args.mode])
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
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    def run(env, agent, seed=None):
        env.reset()
        if seed is not None:
            env.set_seed(seed)
            np.random.seed(seed)
        _, decision_event, done = env.step(None)
        while not done:
            metrics, decision_event, done = env.step(agent.choose_action(decision_event, env.snapshot_list))

        return env.metrics

    print(f"Topologies: {topologies} (total {len(topologies)})")
    print(f"duration: {args.duration}")
    print(f"action mode: {args.mode}")
    for tp in topologies:
        env = Env(scenario="ecr", topology=tp, durations=args.duration)
        results = [tp]
        for i, seed in enumerate(args.seed):
            result = run(env, agent, seed)
            results.extend([result["order_requirements"], result["container_shortage"], result["operation_cost"]])

        with open(file_name, 'a') as fp:
            writer = csv.writer(fp)
            writer.writerow(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("topology", help="supported topologies: toy, global")
    parser.add_argument("mode", help="supported modes: random, null")
    parser.add_argument("-p", "--ports", default="all", help="number of ports")
    parser.add_argument("-l", "--level", default="all", help="difficulty level")
    parser.add_argument("-s", "--seed", type=int, default=[0], action="append", help="random seed")
    parser.add_argument("-d", "--duration", type=int, default=-1, help="duration")
    args = parser.parse_args()
    main(args)
