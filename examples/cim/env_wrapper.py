# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

import numpy as np

from maro.rl import AbsEnvWrapper
from maro.simulator.scenarios.cim.common import Action, ActionType


class CIMEnvWrapper(AbsEnvWrapper):
    def __init__(
        self, env, *, port_attributes, vessel_attributes, num_actions, look_back, max_ports_downstream,
        reward_eval_delay, fulfillment_factor, shortage_factor, time_decay,
        finite_vessel_space=True, has_early_discharge=True 
    ):
        super().__init__(env, reward_eval_delay=reward_eval_delay)
        self.port_attributes = port_attributes
        self.vessel_attributes = vessel_attributes
        self.action_space = list(np.linspace(-1.0, 1.0, num_actions))
        self.look_back = look_back
        self.max_ports_downstream = max_ports_downstream
        self.fulfillment_factor = fulfillment_factor
        self.shortage_factor = shortage_factor
        self.time_decay = time_decay
        self.finite_vessel_space = finite_vessel_space
        self.has_early_discharge = has_early_discharge

    def get_state(self, event):
        vessel_snapshots, port_snapshots = self.env.snapshot_list["vessels"], self.env.snapshot_list["ports"]
        tick, port_idx, vessel_idx = event.tick, event.port_idx, event.vessel_idx
        ticks = [max(0, tick - rt) for rt in range(self.look_back - 1)]
        future_port_idx_list = vessel_snapshots[tick: vessel_idx: 'future_stop_list'].astype('int')
        port_features = port_snapshots[ticks: [port_idx] + list(future_port_idx_list): self.port_attributes]
        vessel_features = vessel_snapshots[tick: vessel_idx: self.vessel_attributes]
        self.state_info = {
            "tick": tick, "action_scope": event.action_scope, "port_idx": port_idx, "vessel_idx": vessel_idx
        }
        state = np.concatenate((port_features, vessel_features))
        return {port_idx: state}

    def get_action(self, action_by_agent):
        vessel_snapshots = self.env.snapshot_list["vessels"]
        action_info = list(action_by_agent.values())[0]
        model_action = action_info[0] if isinstance(action_info, tuple) else action_info
        tick, port, vessel = self.state_info["tick"], self.state_info["port_idx"], self.state_info["vessel_idx"]
        zero_action_idx = len(self.action_space) / 2  # index corresponding to value zero.
        vessel_space = vessel_snapshots[tick:vessel:self.vessel_attributes][2] if self.finite_vessel_space else float("inf")
        early_discharge = vessel_snapshots[tick:vessel:"early_discharge"][0] if self.has_early_discharge else 0
        percent = abs(self.action_space[model_action])

        action_scope = self.state_info["action_scope"]
        if model_action < zero_action_idx:
            action_type = ActionType.LOAD
            actual_action = min(round(percent * action_scope.load), vessel_space)
        elif model_action > zero_action_idx:
            action_type = ActionType.DISCHARGE
            plan_action = percent * (action_scope.discharge + early_discharge) - early_discharge
            actual_action = round(plan_action) if plan_action > 0 else round(percent * action_scope.discharge)
        else:
            actual_action, action_type = 0, None

        return {port: Action(vessel, port, actual_action, action_type)}

    def get_reward(self, tick=None, target_agents=None):
        """Delayed reward evaluation."""
        if tick is None:
            tick = self.env.tick
        port_snapshots = self.env.snapshot_list["ports"]
        start_tick = tick + 1
        ticks = list(range(start_tick, start_tick + self.reward_eval_delay))

        future_fulfillment = port_snapshots[ticks::"fulfillment"]
        future_shortage = port_snapshots[ticks::"shortage"]
        decay_list = [
            self.time_decay ** i for i in range(self.reward_eval_delay)
            for _ in range(future_fulfillment.shape[0] // self.reward_eval_delay)
        ]

        return {
            target_agents[0]: 
            np.float32(
                self.fulfillment_factor * np.dot(future_fulfillment, decay_list) - 
                self.shortage_factor * np.dot(future_shortage, decay_list)
            )
        }
