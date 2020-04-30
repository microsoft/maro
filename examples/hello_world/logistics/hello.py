from maro.simulator import Env
from maro.simulator.scenarios.logistics.common import Action
from random import randint


MAX_EPISODE = 1
MAX_TICKS = 10 # max_episode_length


def start():
    env = Env(scenario="logistics", topology="inventory_placement", start_tick=0, max_tick=MAX_TICKS)

    for episode in range(MAX_EPISODE):
        env.reset()

        # NOTE: first action must be None
        env.step(None)

        rewards = []
        is_done = False
        while not is_done:
            action = Action(0, randint(1, 10))
            reward, decision_event, is_done = env.step(action)
            if not is_done: 
                rewards.append(reward)

    feature_names = ["demand", "stock", "fulfilled", "unfulfilled"]
    features = env.snapshot_list.static_nodes[:0:(feature_names, 0)]
    features = features.reshape(len(env.snapshot_list), len(feature_names)) # one tick one row, features in each row
    print(feature_names)
    print(features)
    print("rewards: {}".format(rewards))
        

if __name__ == "__main__":
    start()
