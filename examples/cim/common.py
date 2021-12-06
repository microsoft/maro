# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

import numpy as np

from maro.rl import Trajectory
from maro.simulator.scenarios.cim.common import Action, ActionType

common_config = {
    "port_attributes": ["empty", "full", "on_shipper", "on_consignee", "booking", "shortage", "fulfillment"],
    "vessel_attributes": ["empty", "full", "remaining_space"],
    "action_space": list(np.linspace(-1.0, 1.0, 21)),
    # Parameters for computing states
    "look_back": 7,
    "max_ports_downstream": 2,
    # Parameters for computing actions
    "finite_vessel_space": True,
    "has_early_discharge": True,
    # Parameters for computing rewards
    "reward_time_window": 99,
    "fulfillment_factor": 1.0,
    "shortage_factor": 1.0,
    "time_decay": 0.97
}


class CIMTrajectory(Trajectory):
    def __init__(
        self, env, *, port_attributes, vessel_attributes, action_space, look_back, max_ports_downstream,
        reward_time_window, fulfillment_factor, shortage_factor, time_decay,
        finite_vessel_space=True, has_early_discharge=True
    ):
        super().__init__(env)
        self.port_attributes = port_attributes
        self.vessel_attributes = vessel_attributes
        self.action_space = action_space
        self.look_back = look_back
        self.max_ports_downstream = max_ports_downstream
        self.reward_time_window = reward_time_window
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
        return {port_idx: np.concatenate((port_features, vessel_features))}

    def get_action(self, action_by_agent, event):
        vessel_snapshots = self.env.snapshot_list["vessels"]
        action_info = list(action_by_agent.values())[0]
        model_action = action_info[0] if isinstance(action_info, tuple) else action_info
        scope, tick, port, vessel = event.action_scope, event.tick, event.port_idx, event.vessel_idx
        zero_action_idx = len(self.action_space) / 2  # index corresponding to value zero.
        vessel_space = vessel_snapshots[tick:vessel:self.vessel_attributes][2] if self.finite_vessel_space else float("inf")
        early_discharge = vessel_snapshots[tick:vessel:"early_discharge"][0] if self.has_early_discharge else 0
        percent = abs(self.action_space[model_action])

        if model_action < zero_action_idx:
            action_type = ActionType.LOAD
            actual_action = min(round(percent * scope.load), vessel_space)
        elif model_action > zero_action_idx:
            action_type = ActionType.DISCHARGE
            plan_action = percent * (scope.discharge + early_discharge) - early_discharge
            actual_action = round(plan_action) if plan_action > 0 else round(percent * scope.discharge)
        else:
            actual_action, action_type = 0, ActionType.LOAD

        return {port: Action(vessel, port, actual_action, action_type)}

    def get_offline_reward(self, event):
        port_snapshots = self.env.snapshot_list["ports"]
        start_tick = event.tick + 1
        ticks = list(range(start_tick, start_tick + self.reward_time_window))

        future_fulfillment = port_snapshots[ticks::"fulfillment"]
        future_shortage = port_snapshots[ticks::"shortage"]
        decay_list = [
            self.time_decay ** i for i in range(self.reward_time_window)
            for _ in range(future_fulfillment.shape[0] // self.reward_time_window)
        ]

        tot_fulfillment = np.dot(future_fulfillment, decay_list)
        tot_shortage = np.dot(future_shortage, decay_list)

        return np.float32(self.fulfillment_factor * tot_fulfillment - self.shortage_factor * tot_shortage)

    def on_env_feedback(self, event, state_by_agent, action_by_agent, reward):
        self.trajectory["event"].append(event)
        self.trajectory["state"].append(state_by_agent)
        self.trajectory["action"].append(action_by_agent)
