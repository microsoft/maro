

import math
import os
import random
import shutil

import yaml

TOPOLOGY_LIST = ["toy.4p_ssdd", "toy.5p_ssddd", "toy.6p_sssbdd", "global_trade.22p"]
SAILING_TIME = 7
VESSEL_CAPACITY_REDUNDANCY_RATIOS = [20, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0, 2.5, 2.5]
VESSEL_CAPACITY_DELTA_RATIO = 0.1
AVG_ORDER_RATIO = 0.02
ORDER_RATIO_DELTA = 0.005
ORDER_NOISE = 0.002
PERIOD = 112


def generate_noise(value, min_prop, max_prop):
    return value * random.uniform(min_prop, max_prop)


def save_new_topology(src: str):
    src_dir = src + "_l0.0/config.yml"
    with open(src_dir, "r") as f:
        src_dict = yaml.safe_load(f)
    src_png_path = src + "_l0.0/topology.png"

    def save_new_level(level: int, config_dict: dict):
        change_vessel_capacity(level)
        new_dict = src + "_l0." + str(level)
        os.makedirs(new_dict, exist_ok=True)
        with open(new_dict + "/config.yml", "w") as dump_file:
            yaml.safe_dump(config_dict, dump_file)
        if level == 0:
            return
        if os.path.exists(src_png_path):
            shutil.copyfile(src_png_path, new_dict + "/topology.png")

    def change_vessel_capacity(level):
        for vessel in src_dict["vessels"].values():
            route_proportion = route_proportions[vessel['route']['route_name']]
            vessel["capacity"] = int(
                AVG_ORDER_RATIO * route_proportion * SAILING_TIME * total_containers
                * VESSEL_CAPACITY_REDUNDANCY_RATIOS[level]
            )

    src_dict['container_usage_proportion']['period'] = PERIOD
    src_dict['container_usage_proportion']['sample_nodes'] = [[0, AVG_ORDER_RATIO], [PERIOD - 1, AVG_ORDER_RATIO]]

    total_containers = src_dict['total_containers']
    route_proportions = {route_name: 0 for route_name in src_dict["routes"].keys()}
    ports_in_routes = {
        route_name: [stop['port_name'] for stop in src_dict['routes'][route_name]]
        for route_name in src_dict["routes"].keys()
    }
    for source_port_name, port in src_dict["ports"].items():
        if 'targets' in port['order_distribution'].keys():
            source_proportion = port['order_distribution']['source']['proportion']
            for target_port_name, target_proportion in port['order_distribution']['targets'].items():
                for route_name, port_list in ports_in_routes.items():
                    if source_port_name in port_list and target_port_name in port_list:
                        route_proportions[route_name] += source_proportion * target_proportion['proportion']
                        break

    save_new_level(0, src_dict)

    save_new_level(1, src_dict)

    for i, vessel in enumerate(src_dict["vessels"].values()):
        vessel["capacity"] += int(vessel["capacity"] * VESSEL_CAPACITY_DELTA_RATIO * (i % 3 - 1))
    save_new_level(2, src_dict)

    sine_distribution = [
        [i, AVG_ORDER_RATIO - ORDER_RATIO_DELTA * math.cos(i / (PERIOD // 2) * math.pi)] for i in range(PERIOD)
    ]
    src_dict['container_usage_proportion']['sample_nodes'] = sine_distribution
    save_new_level(3, src_dict)

    src_dict['container_usage_proportion']["sample_noise"] = ORDER_NOISE
    for port in src_dict["ports"].values():
        order_distribution = port["order_distribution"]
        source_proportion = order_distribution["source"]["proportion"]
        order_distribution["source"]["noise"] = generate_noise(source_proportion, 0, 0.2)
        if "targets" in order_distribution:
            for target in order_distribution["targets"].values():
                target_proportion = target["proportion"]
                target["noise"] = generate_noise(target_proportion, 0, 0.2)
    save_new_level(4, src_dict)

    for port in src_dict["ports"].values():
        full_return = port["full_return"]["buffer_ticks"]
        port["full_return"]["noise"] = math.ceil(generate_noise(full_return, 0, 0.5))
        empty_return = port["empty_return"]["buffer_ticks"]
        port["empty_return"]["noise"] = math.ceil(generate_noise(empty_return, 0, 0.5))
    save_new_level(5, src_dict)

    for vessel in src_dict["vessels"].values():
        speed = vessel["sailing"]["speed"]
        vessel["sailing"]["noise"] = math.ceil(generate_noise(speed, 0, 0.2))
        duration = vessel["parking"]["duration"]
        vessel["parking"]["noise"] = math.ceil(generate_noise(duration, 0, 0.5))
    save_new_level(6, src_dict)

    for i, vessel in enumerate(src_dict["vessels"].values()):
        vessel["sailing"]["speed"] = int(vessel["sailing"]["speed"] * (10 - i % 3) / 10)
    save_new_level(7, src_dict)

    sine_fluctuate = [[i, abs(math.cos(i / (PERIOD // 8) * math.pi))] for i in range(PERIOD // 4)]
    valley = AVG_ORDER_RATIO - ORDER_RATIO_DELTA
    sine_fluc = sine_fluctuate
    sine_dist = sine_distribution
    multi_sine_distribution = [
        [i, sine_fluc[i % (PERIOD // 4)][1] * (sine_dist[i][1] - valley) * math.pi / 2 + valley] for i in range(PERIOD)
    ]
    src_dict['container_usage_proportion']['sample_nodes'] = multi_sine_distribution
    save_new_level(8, src_dict)


if __name__ == "__main__":
    for topology in TOPOLOGY_LIST:
        save_new_topology(topology)
