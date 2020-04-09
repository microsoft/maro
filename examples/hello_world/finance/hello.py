from maro.simulator import Env


env = Env("finance", "test", max_tick=10)

reward, decision_event, is_done = env.step(None)

while not is_done:
    reward, decision_event, is_done = env.step(None)

