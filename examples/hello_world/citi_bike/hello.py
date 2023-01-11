# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator import Env
from maro.simulator.scenarios.citi_bike.common import Action, DecisionEvent

auto_event_mode = False
start_tick = 0
durations = 100
max_ep = 2

opts = dict()
"""
enable-dump-snapshot parameter means business_engine needs dump snapshot data before reset.
If you leave value to empty string, it will dump to current folder.
For getting dump data, please uncomment below line and specify dump destination folder.
"""
# opts['enable-dump-snapshot'] = ''

env = Env(
    scenario="citi_bike",
    topology="toy.4s_4t",
    start_tick=start_tick,
    durations=durations,
    snapshot_resolution=60,
    options=opts,
)

print(env.summary)

for ep in range(max_ep):
    metrics = None
    decision_evt: DecisionEvent = None
    is_done = False
    action = None

    while not is_done:
        metrics, decision_evt, is_done = env.step(action)

        # It will be None at the end.
        if decision_evt is not None:
            action = Action(decision_evt.station_idx, 0, 10)

    station_ss = env.snapshot_list["stations"]
    shortage_states = station_ss[::"shortage"]
    print("total shortage", shortage_states.sum())

    trips_states = station_ss[::"trip_requirement"]
    print("total trip", trips_states.sum())

    cost_states = station_ss[::["extra_cost", "transfer_cost"]]

    print("total cost", cost_states.sum())

    matrix_ss = env.snapshot_list["matrices"]

    # Since we may have different snapshot resolution,
    # so we should use frame_index to retrieve index in snapshots of current tick.
    last_snapshot_index = env.frame_index

    # NOTE: We have not clear the trip adj at each tick so it is an accumulative value,
    # then we can just query last snapshot to calc total trips.
    trips_adj = matrix_ss[last_snapshot_index::"trips_adj"]

    # Reshape it we need an easy way to access.
    # trips_adj = trips_adj.reshape((-1, len(station_ss)))

    print("total trips from trips adj", trips_adj.sum())

    fulfillments = station_ss[::"fulfillment"]

    env.reset()
