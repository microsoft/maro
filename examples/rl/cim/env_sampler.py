# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np

from maro.rl.learning import EnvSampler
from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action, ActionType

cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)

from callbacks import post_step
from config import action_shaping_conf, env_conf, reward_shaping_conf, state_shaping_conf, vessel_features
from policies import policy_func_dict


def get_state(env, event, state_shaping_conf):
    vessel_snapshots, port_snapshots = env.snapshot_list["vessels"], env.snapshot_list["ports"]
    port_idx, vessel_idx = event.port_idx, event.vessel_idx
    ticks = [max(0, env.tick - rt) for rt in range(state_shaping_conf["look_back"] - 1)]
    future_port_list = vessel_snapshots[env.tick: vessel_idx: 'future_stop_list'].astype('int')
    port_features = port_snapshots[ticks: [port_idx] + list(future_port_list): state_shaping_conf["port_features"]]
    vessel_features = vessel_snapshots[env.tick: vessel_idx: state_shaping_conf["vessel_features"]]
    state = np.concatenate((port_features, vessel_features))
    return {port_idx: state}


def get_env_actions(action_by_agent, env, event, action_shaping_conf):
    port_idx, action = list(action_by_agent.items()).pop()
    vessel_idx, action_scope = event.vessel_idx, event.action_scope
    vessel_snapshots = env.snapshot_list["vessels"]
    vessel_space = (
        vessel_snapshots[env.tick:vessel_idx:vessel_features][2]
        if action_shaping_conf["finite_vessel_space"] else float("inf")
    )

    model_action = action["action"] if isinstance(action, dict) else action
    percent = abs(action_shaping_conf["action_space"][model_action])
    zero_action_idx = len(action_shaping_conf["action_space"]) / 2  # index corresponding to value zero.
    if model_action < zero_action_idx:
        action_type = ActionType.LOAD
        actual_action = min(round(percent * action_scope.load), vessel_space)
    elif model_action > zero_action_idx:
        action_type = ActionType.DISCHARGE
        if action_shaping_conf["has_early_discharge"]:
            early_discharge = vessel_snapshots[env.tick:vessel_idx:"early_discharge"][0]
        else:
            early_discharge = 0
        plan_action = percent * (action_scope.discharge + early_discharge) - early_discharge
        actual_action = round(plan_action) if plan_action > 0 else round(percent * action_scope.discharge)
    else:
        actual_action, action_type = 0, None

    return [Action(port_idx=port_idx, vessel_idx=vessel_idx, quantity=actual_action, action_type=action_type)]


def get_reward(env, actions, tick, reward_shaping_conf):
    """Delayed reward evaluation."""
    start_tick = tick + 1
    ticks = list(range(start_tick, start_tick + reward_shaping_conf["time_window"]))

    # Get the ports that took actions at the given tick
    ports = [action.port_idx for action in actions]
    port_snapshots = env.snapshot_list["ports"]
    future_fulfillment = port_snapshots[ticks:ports:"fulfillment"].reshape(len(ticks), -1)
    future_shortage = port_snapshots[ticks:ports:"shortage"].reshape(len(ticks), -1)

    decay_list = [reward_shaping_conf["time_decay"] ** i for i in range(reward_shaping_conf["time_window"])]
    rewards = np.float32(
        reward_shaping_conf["fulfillment_factor"] * np.dot(future_fulfillment.T, decay_list)
        - reward_shaping_conf["shortage_factor"] * np.dot(future_shortage.T, decay_list)
    )
    return {agent_id: reward for agent_id, reward in zip(ports, rewards)}


agent2policy = {
    0: "ac.0",
    1: "ac.1",
    2: "dqn",
    3: "ac.2"
}


def get_env_sampler():
    return EnvSampler(
        lambda: Env(**env_conf),
        policy_func_dict,
        agent2policy,
        get_state,
        get_env_actions,
        get_reward,
        reward_eval_delay=reward_shaping_conf["reward_eval_delay"],
        post_step=post_step,
        state_shaping_conf=state_shaping_conf,
        action_shaping_conf=action_shaping_conf,
        reward_shaping_conf=reward_shaping_conf
    )
