import numpy as np
from maro.simulator.frame import FrameNodeType
from maro.simulator.core import Env
from maro.simulator.scenarios.bike.common import Action

max_ticks = 2*24
total_ep = 2

# 48 ticks (hours), 60 units (minutes) per tick
env = Env("bike", "test", max_ticks, tick_units=60)

for i in range(total_ep):
    env.reset()

    reward, decision_event, is_done = env.step(None)

    while not is_done:
        reward, decision_event, is_done = env.step(Action(0, 1, 0))

print("snapshot list length", len(env.snapshot_list))
inv = env.snapshot_list.static_nodes[::("bikes", 0)]

inv = inv.reshape(max_ticks, len(env.agent_idx_list)) # max_ticks, cells

print(f"bike number at cell 0 (index not id) in {max_ticks} hours.")
print(inv[:, 0]) # 1st column means bikes station of station 0 at all the ticks

# features for cell index: 0
feature_names = ["bikes", "shortage", "trip_requirement", "temperature", "weather", "holiday"]
features = env.snapshot_list.static_nodes[:0:(feature_names, 0)]
features = features.reshape(max_ticks, len(feature_names)) # one tick one row, features in each row
print(feature_names)
print(features)

#neighbors
#NOTE: since the neighbors will not change, so we can just query 1st tick
# -1 means invalid cell id
neighbors = env.snapshot_list.static_nodes[0::("neighbors", [0, 1, 2, 3, 4, 5])]
neighbors = neighbors.reshape(len(env.agent_idx_list), 6)
print("neighbors")
print(neighbors)

# extra cost
extra_cost = env.snapshot_list.static_nodes[::("extra_cost", 0)]
extra_cost = extra_cost.reshape(max_ticks, len(env.agent_idx_list))
print("extra cost for cells in all ticks")
print(extra_cost)

# NOTE: code to check if there is any extra cost
# for i, r in enumerate(extra_cost):
#     non_zero_extra_cost_cell_index = np.where(r>0)[0]

#     if len(non_zero_extra_cost_cell_index):
#         # print("tick", i, non_zero_extra_cost_cell_index)
#         print("tick", i, non_zero_extra_cost_cell_index)
#         print(r[r>0])
