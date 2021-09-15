# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np

from maro.rl.learning import AbsEnvSampler
from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action, ActionType

cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)

from callbacks import post_step
from config import (
    action_shaping_conf, env_conf, port_attributes, reward_shaping_conf, state_shaping_conf, vessel_attributes
)
from policies import policy_func_dict


class CIMEnvSampler(AbsEnvSampler):
    def get_state(self, tick=None):
        if tick is None:
            tick = self.env.tick
        vessel_snapshots, port_snapshots = self.env.snapshot_list["vessels"], self.env.snapshot_list["ports"]
        port_idx, vessel_idx = self.event.port_idx, self.event.vessel_idx
        ticks = [max(0, tick - rt) for rt in range(state_shaping_conf["look_back"] - 1)]
        future_port_list = vessel_snapshots[tick: vessel_idx: 'future_stop_list'].astype('int') 
        state = np.concatenate([
            port_snapshots[ticks : [port_idx] + list(future_port_list) : port_attributes],
            vessel_snapshots[tick : vessel_idx : vessel_attributes]
        ])
        return {port_idx: state}

    def get_env_actions(self, action_by_agent):
        action_space = action_shaping_conf["action_space"]
        finite_vsl_space = action_shaping_conf["finite_vessel_space"]
        has_early_discharge = action_shaping_conf["has_early_discharge"]

        port_idx, action = list(action_by_agent.items()).pop()
        vsl_idx, action_scope = self.event.vessel_idx, self.event.action_scope
        vsl_snapshots = self.env.snapshot_list["vessels"]
        vsl_space = vsl_snapshots[self.env.tick:vsl_idx:vessel_attributes][2] if finite_vsl_space else float("inf")

        model_action = action["action"] if isinstance(action, dict) else action    
        percent = abs(action_space[model_action])
        zero_action_idx = len(action_space) / 2  # index corresponding to value zero.
        if model_action < zero_action_idx:
            action_type = ActionType.LOAD
            actual_action = min(round(percent * action_scope.load), vsl_space)
        elif model_action > zero_action_idx:
            action_type = ActionType.DISCHARGE
            early_discharge = vsl_snapshots[self.env.tick:vsl_idx:"early_discharge"][0] if has_early_discharge else 0
            plan_action = percent * (action_scope.discharge + early_discharge) - early_discharge
            actual_action = round(plan_action) if plan_action > 0 else round(percent * action_scope.discharge)
        else:
            actual_action, action_type = 0, None

        return [Action(port_idx=port_idx, vessel_idx=vsl_idx, quantity=actual_action, action_type=action_type)]

    def get_reward(self, actions, tick):
        """Delayed reward evaluation."""
        start_tick = tick + 1
        ticks = list(range(start_tick, start_tick + reward_shaping_conf["time_window"]))

        # Get the ports that took actions at the given tick
        ports = [action.port_idx for action in actions]
        port_snapshots = self.env.snapshot_list["ports"]
        future_fulfillment = port_snapshots[ticks:ports:"fulfillment"].reshape(len(ticks), -1)
        future_shortage = port_snapshots[ticks:ports:"shortage"].reshape(len(ticks), -1)

        decay_list = [reward_shaping_conf["time_decay"] ** i for i in range(reward_shaping_conf["time_window"])]
        rewards = np.float32(
            reward_shaping_conf["fulfillment_factor"] * np.dot(future_fulfillment.T, decay_list)
            - reward_shaping_conf["shortage_factor"] * np.dot(future_shortage.T, decay_list)
        )
        return {agent_id: reward for agent_id, reward in zip(ports, rewards)}


agent2policy = {agent: f"ac.{agent}" for agent in Env(**env_conf).agent_idx_list}

def get_env_sampler():
    return CIMEnvSampler(
        get_env=lambda: Env(**env_conf),
        get_policy_func_dict=policy_func_dict,
        agent2policy=agent2policy,
        reward_eval_delay=reward_shaping_conf["time_window"],
        post_step=post_step,
        policies_to_parallelize=[]
    )
