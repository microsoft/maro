from maro.simulator.core import Env

env = Env("bike", "test", 48)

reward, decision_event, is_done = env.step(None)

while not is_done:
    reward, decision_event, is_done = env.step(None)
