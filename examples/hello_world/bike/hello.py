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
        decision_event.action_scope

        reward, decision_event, is_done = env.step(Action(0, 0, 0))

    print("snapshot list length", len(env.snapshot_list))
    inv = env.snapshot_list.static_nodes[::("bikes", 0)]

    inv = inv.reshape(max_ticks, 52) # max_ticks, 52 stations

    print(f"bike number at station 0 (index not id) in {max_ticks} hours.")
    print(inv[:, 0]) # 1st column means bikes station of station 0 at all the ticks