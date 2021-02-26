# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

import numpy as np

from maro.simulator.scenarios.cim.common import Action, ActionType

PORT_ATTRIBUTES = ["empty", "full", "on_shipper", "on_consignee", "booking", "shortage", "fulfillment"]
VESSEL_ATTRIBUTES = ["empty", "full", "remaining_space"]
ACTION_SPACE = list(np.linspace(-1.0, 1.0, 21))

# Parameters for computing states
LOOK_BACK = 7
MAX_PORTS_DOWNSTREAM = 2

# Parameters for computing rewards
REWARD_TIME_WINDOW = 100
FULFILLMENT_FACTOR = 1.0
SHORTAGE_FACTOR = 1.0
TIME_DECAY = 0.97


def get_state(decision_event, snapshots, look_back=LOOK_BACK):
    tick, port_idx, vessel_idx = decision_event.tick, decision_event.port_idx, decision_event.vessel_idx
    ticks = [tick - rt for rt in range(look_back - 1)]
    future_port_idx_list = snapshots["vessels"][tick: vessel_idx: 'future_stop_list'].astype('int')
    port_features = snapshots["ports"][ticks: [port_idx] + list(future_port_idx_list): PORT_ATTRIBUTES]
    vessel_features = snapshots["vessels"][tick: vessel_idx: VESSEL_ATTRIBUTES]
    return np.concatenate((port_features, vessel_features))


def get_env_action(
    model_action, decision_event, vessel_snapshots, 
    action_space=ACTION_SPACE, finite_vessel_space=True, has_early_discharge=True
):
    scope = decision_event.action_scope
    tick = decision_event.tick
    port = decision_event.port_idx
    vessel = decision_event.vessel_idx

    zero_action_idx = len(action_space) / 2  # index corresponding to value zero.

    vessel_space = vessel_snapshots[tick:vessel:VESSEL_ATTRIBUTES][2] if finite_vessel_space else float("inf")
    early_discharge = vessel_snapshots[tick:vessel:"early_discharge"][0] if has_early_discharge else 0
    percent = abs(action_space[model_action])

    if model_action < zero_action_idx:
        action_type = ActionType.LOAD
        actual_action = min(round(percent * scope.load), vessel_space)
    elif model_action > zero_action_idx:
        action_type = ActionType.DISCHARGE
        plan_action = percent * (scope.discharge + early_discharge) - early_discharge
        actual_action = round(plan_action) if plan_action > 0 else round(percent * scope.discharge)
    else:
        actual_action, action_type = 0, None

    return Action(vessel, port, actual_action, action_type)


def get_reward(
    decision_event, port_snapshots, reward_time_window=REWARD_TIME_WINDOW, time_decay=TIME_DECAY,
    fulfillment_factor=FULFILLMENT_FACTOR, shortage_factor=SHORTAGE_FACTOR    
):
    start_tick = decision_event.tick + 1
    end_tick = decision_event.tick + reward_time_window
    ticks = list(range(start_tick, end_tick))

    future_fulfillment = port_snapshots[ticks::"fulfillment"]
    future_shortage = port_snapshots[ticks::"shortage"]
    decay_list = [
        time_decay ** i for i in range(end_tick - start_tick)
        for _ in range(future_fulfillment.shape[0] // (end_tick - start_tick))
    ]

    tot_fulfillment = np.dot(future_fulfillment, decay_list)
    tot_shortage = np.dot(future_shortage, decay_list)

    return np.float32(fulfillment_factor * tot_fulfillment - shortage_factor * tot_shortage)
