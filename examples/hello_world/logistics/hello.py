from maro.simulator import Env
from random import randint

MAX_EPISODE = 1
MAX_TICKS = 1000 # max_episode_length

def start():
    env = Env(scenario="logistics", topology="sample", start_tick=0, max_tick=MAX_TICKS)
    reward = None
    decision_event = None
    is_done = False

    for episode in range(MAX_EPISODE):
        env.reset()

        reward, decision_event, is_done = env.step(None)

        while not is_done:
            reward, decision_event, is_done = env.step(randint(1, 10))

        # following code show how to retrieve states from snapshotlist
        stock_at_all_tick = env.snapshot_list.static_nodes[::("stock", 0)]

        print(stock_at_all_tick)

        # since our business engine will stop before reach the max tick, so we should use current tick as upper bound to query valid sequence
        ticks = [i for i in range(env.tick+1)] # include last tick

        stock_to_the_end = env.snapshot_list.static_nodes[ticks::("stock", 0)]

        print(stock_to_the_end)

if __name__ == "__main__":
    start()