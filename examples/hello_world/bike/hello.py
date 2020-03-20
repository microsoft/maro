from maro.simulator.core import Env
from maro.simulator.scenarios.bike.common import Action

# 48 ticks (hours), 60 units (minutes) per tick
env = Env("bike", "test", 2*24, tick_units=60)

reward, decision_event, is_done = env.step(None)

while not is_done:
    decision_event.action_scope

    reward, decision_event, is_done = env.step(Action(0, 0, 0))

print("snapshot list length", len(env.snapshot_list))
inv = env.snapshot_list.static_nodes[:0:("bikes", 0)]
print(inv)