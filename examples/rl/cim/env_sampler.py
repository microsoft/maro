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
from config import env_conf, env_sampler_conf
from policies import policy_func_dict


class CIMEnvSampler(AbsEnvSampler):
    def __init__(
        self, get_env, get_policy_func_dict, agent2policy, get_eval_env=None, reward_eval_delay=99, post_step=None
    ):
        super().__init__(
            get_env, get_policy_func_dict, agent2policy,
            get_eval_env=get_eval_env, reward_eval_delay=reward_eval_delay, post_step=post_step
        )
        self.action_space = list(np.linspace(-1.0, 1.0, env_sampler_conf["num_actions"]))
        self._last_action_tick = None
        self._state_info = None

    def get_state(self, event, tick=None):
        if tick is None:
            tick = self.env.tick
        vessel_snapshots, port_snapshots = self.env.snapshot_list["vessels"], self.env.snapshot_list["ports"]
        port_idx, vessel_idx = event.port_idx, event.vessel_idx
        ticks = [max(0, tick - rt) for rt in range(env_sampler_conf["look_back"] - 1)]
        future_port_list = vessel_snapshots[tick: vessel_idx: 'future_stop_list'].astype('int')
        port_features = port_snapshots[ticks: [port_idx] + list(future_port_list): env_sampler_conf["port_features"]]
        vessel_features = vessel_snapshots[tick: vessel_idx: env_sampler_conf["vessel_features"]]
        self._state_info = {"tick": tick, "action_scope": event.action_scope, "vessel_idx": vessel_idx}
        state = np.concatenate((port_features, vessel_features))
        self._last_action_tick = tick
        return {port_idx: state}

    def to_env_action(self, action_by_agent: dict):
        port_idx, action = list(action_by_agent.items()).pop()
        tick = self._state_info["tick"]
        vessel = self._state_info["vessel_idx"]
        action_scope = self._state_info["action_scope"]
        vessel_snapshots = self.env.snapshot_list["vessels"]
        vessel_space = (
            vessel_snapshots[tick:vessel:env_sampler_conf["vessel_features"]][2]
            if env_sampler_conf["finite_vessel_space"] else float("inf")
        )

        model_action = action["action"] if isinstance(action, dict) else action
        percent = abs(self.action_space[model_action])
        zero_action_idx = len(self.action_space) / 2  # index corresponding to value zero.
        if model_action < zero_action_idx:
            action_type = ActionType.LOAD
            actual_action = min(round(percent * action_scope.load), vessel_space)
        elif model_action > zero_action_idx:
            action_type = ActionType.DISCHARGE
            if env_sampler_conf["has_early_discharge"]:
                early_discharge = vessel_snapshots[tick:vessel:"early_discharge"][0]
            else:
                early_discharge = 0
            plan_action = percent * (action_scope.discharge + early_discharge) - early_discharge
            actual_action = round(plan_action) if plan_action > 0 else round(percent * action_scope.discharge)
        else:
            actual_action, action_type = 0, None

        return [Action(port_idx=port_idx, vessel_idx=vessel, quantity=actual_action, action_type=action_type)]

    def get_reward(self, actions, tick=None):
        """Delayed reward evaluation."""
        if tick is None:
            tick = self._last_action_tick
        start_tick = tick + 1
        ticks = list(range(start_tick, start_tick + self.reward_eval_delay))

        # Get the ports that took actions at the given tick
        ports = [action.port_idx for action in actions]
        port_snapshots = self.env.snapshot_list["ports"]
        future_fulfillment = port_snapshots[ticks:ports:"fulfillment"].reshape(len(ticks), -1)
        future_shortage = port_snapshots[ticks:ports:"shortage"].reshape(len(ticks), -1)

        decay_list = [env_sampler_conf["time_decay"] ** i for i in range(self.reward_eval_delay)]
        rewards = np.float32(
            env_sampler_conf["fulfillment_factor"] * np.dot(future_fulfillment.T, decay_list)
            - env_sampler_conf["shortage_factor"] * np.dot(future_shortage.T, decay_list)
        )
        return {agent_id: reward for agent_id, reward in zip(ports, rewards)}


agent2policy = {
    0: "ac.0",
    1: "ac.1",
    2: "dqn",
    3: "ac.2"
}


def get_env_sampler():
    return CIMEnvSampler(
        lambda: Env(**env_conf),
        policy_func_dict,
        agent2policy,
        reward_eval_delay=env_sampler_conf["reward_eval_delay"],
        post_step=post_step
    )
