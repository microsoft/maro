from maro.simulator.core import Env

# 48 ticks (hours), 60 units (minutes) per tick
env = Env("bike", "test", 10*24, tick_units=60)

reward, decision_event, is_done = env.step(None)

while not is_done:
    reward, decision_event, is_done = env.step(None)
