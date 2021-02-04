import ast
import io
import os
import random
import shutil
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import yaml

from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action
from maro.utils import clone, convert_dottable


def compute_v2p_degree_matrix(env):
    """This function compute the adjacent matrix."""
    topo_config = env.configs
    static_dict = env.summary["node_mapping"]["ports"]
    dynamic_dict = env.summary["node_mapping"]["vessels"]
    adj_matrix = np.zeros((len(dynamic_dict), len(static_dict)), dtype=np.int)
    for v, vinfo in topo_config["vessels"].items():
        route_name = vinfo["route"]["route_name"]
        route = topo_config["routes"][route_name]
        vid = dynamic_dict[v]
        for p in route:
            adj_matrix[vid][static_dict[p["port_name"]]] += 1

    return adj_matrix


def log2json(file_path):
    """load the log file as a json list."""
    with open(file_path, "r") as fp:
        lines = fp.read().splitlines()
        json_list = "[" + ",".join(lines) + "]"
        return ast.literal_eval(json_list)


def decision_cnt_analysis(env, pv=False, buffer_size=8):
    if not pv:
        decision_cnt = {str(agent_id): buffer_size for agent_id in env.agent_idx_list}
        r, pa, is_done = env.step(None)
        while not is_done:
            decision_cnt[str(pa.port_idx)] += 1
            action = Action(pa.vessel_idx, pa.port_idx, 0)
            r, pa, is_done = env.step(action)
    else:
        decision_cnt = OrderedDict()
        r, pa, is_done = env.step(None)
        while not is_done:
            if (pa.port_idx, pa.vessel_idx) not in decision_cnt:
                decision_cnt[pa.port_idx, pa.vessel_idx] = buffer_size
            else:
                decision_cnt[pa.port_idx, pa.vessel_idx] += 1
            action = Action(pa.vessel_idx, pa.port_idx, 0)
            r, pa, is_done = env.step(action)
    env.reset()
    return decision_cnt


def random_shortage(env, tick, action_dim=21):
    _, pa, is_done = env.step(None)
    node_cnt = len(env.summary["node_mapping"]["ports"])
    while not is_done:
        """
        load, discharge = pa.action_scope.load, pa.action_scope.discharge
        action_idx = np.random.randint(action_dim) - zero_idx
        if action_idx < 0:
            actual_action = int(1.0*action_idx/zero_idx*load)
        else:
            actual_action = int(1.0*action_idx/zero_idx*discharge)
        """
        action = Action(pa.vessel_idx, pa.port_idx, 0)
        r, pa, is_done = env.step(action)

    shs = env.snapshot_list["ports"][tick - 1:list(range(node_cnt)):"acc_shortage"]
    fus = env.snapshot_list["ports"][tick - 1:list(range(node_cnt)):"acc_fulfillment"]
    env.reset()
    return fus - shs, np.sum(shs + fus)


def return_scaler(env, tick, gamma, action_dim=21):
    R, tot_amount = random_shortage(env, tick, action_dim)
    Rs_mean = np.mean(R) / tick / (1 - gamma)
    return abs(1.0 / Rs_mean), tot_amount


def load_config(config_pth):
    with io.open(config_pth, "r") as in_file:
        raw_config = yaml.safe_load(in_file)
        config = convert_dottable(raw_config)

    if config.env.seed < 0:
        config.env.seed = random.randint(0, 99999)

    regularize_config(config)
    return config


def save_config(config, config_pth):
    with open(config_pth, "w") as fp:
        config = dottable2dict(config)
        config["agent"]["exp_per_ep"] = [f"{k[0]}, {k[1]}, {d}" for k, d in config["agent"]["exp_per_ep"].items()]
        yaml.safe_dump(config, fp)


def dottable2dict(config):
    if isinstance(config, float):
        return str(config)
    if not isinstance(config, dict):
        return clone(config)
    rt = {}
    for k, v in config.items():
        rt[k] = dottable2dict(v)
    return rt


def save_code(folder, save_pth):
    save_path = os.path.join(save_pth, "code")
    code_pth = os.path.join(os.getcwd(), folder)
    shutil.copytree(code_pth, save_path)


def fix_seed(env, seed):
    env.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def zero_play(**args):
    env = Env(**args)
    _, pa, is_done = env.step(None)
    while not is_done:
        action = Action(pa.vessel_idx, pa.port_idx, 0)
        r, pa, is_done = env.step(action)
    return env.snapshot_list


def regularize_config(config):
    def parse_value(v):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                if v == "false" or v == "False":
                    return False
                elif v == "true" or v == "True":
                    return True
                else:
                    return v

    def set_attr(config, attrs, value):
        if len(attrs) == 1:
            config[attrs[0]] = value
        else:
            set_attr(config[attrs[0]], attrs[1:], value)

    all_args = sys.argv[1:]
    for i in range(len(all_args) // 2):
        name = all_args[i * 2]
        attrs = name[2:].split(".")
        value = parse_value(all_args[i * 2 + 1])
        set_attr(config, attrs, value)


def analysis_speed(env):
    speed_dict = defaultdict(int)
    eq_speed = 0
    for ves in env.configs["vessels"].values():
        speed_dict[ves["sailing"]["speed"]] += 1
    for sp, cnt in speed_dict.items():
        eq_speed += 1.0 * cnt / sp
    eq_speed = 1.0 / eq_speed
    return speed_dict, eq_speed
