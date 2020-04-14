"""
This is hello world for sample scenario, used to show how to customzie a scenario
"""

## STEP 5: use the new scenario

from maro.simulator.core import Env
from maro.simulator.utils.random import random as sim_random
from maro.simulator.scenarios.sample.common import DecisionEvent, Action

EPISODE_NUM = 2
MAX_TICK = 20

def start():
    # create our env first
    # scenario: folder name of our new scenario
    # topology: folder name of config under scenario/topologies
    env = Env(scenario="sample", topology="sample", start_tick=0, max_tick=MAX_TICK, frame_resolution=1)

    reward = None # reward from simulator
    decision_event: DecisionEvent = None # decision event from simulator to ask an action
    is_done: bool = False # if simulator reach the end

    action = Action(0, 1111) # our dummy action

    for ep in range(EPISODE_NUM):
        # reset our env first
        env.reset()

        # NOTE: we must pass None at first step at each episode
        reward, decision_event, is_done = env.step(None)

        while not is_done:
            reward, decision_event, is_done = env.step(action)

    # NOTE: use the snapshot_list before env.reset(), or you can only get the initial value
    # usage of snapshot_list
    
    # you can access snapshot_list with slice interface: snapshot_list.static_node/dynamic_node
    # NOTE: snapshot_list only support node index, not id or other identifier

    # slice interface accept 3 parameters
    # 1. ticks: tick of states to query, can be None, means all the ticks (to current tick)
    # 2. node index list: list of node index to query, can be None, means all the nodes
    # 3. tuple of attribute list: attribute name and slot index, like ([attribute name list], [slot list])

    # all the value of static node attribute "a" (1st slot) at all the ticks
    a_for_all_ticks = env.snapshot_list.static_nodes[::("a", 0)]

    # the output is 1d numpy array, with size = len(ticks) * len(nodes) * len(attributes)
    # so we can reshape it as following, then each row is values of static nodes at that tick
    a_for_all_ticks = a_for_all_ticks.reshape((MAX_TICK, -1))

    print("a at all the ticks")
    print(a_for_all_ticks)

    # attributes b (2nd slot)for 1st static node at all the ticks
    b_at_all_ticks = env.snapshot_list.static_nodes[:0:("b", 1)]

    print("b (2nd slot) at all the ticks")
    print(b_at_all_ticks)

    # attribute c of 1st node at tick 1
    c_at_specified_tick = env.snapshot_list.dynamic_nodes[1:0:("c", 0)]

    print("attribute c at tick 1")
    print(c_at_specified_tick)

    # value at 1st slot for attribute "a" and "b" at all the ticks
    a_b = env.snapshot_list.static_nodes[::(["a", "b"], 0)]
    a_b = a_b.reshape((MAX_TICK, -1))

    print("attribute 'a' and 'b' for all the nodes")
    print(a_b)

    # this is value of attribute 'a'
    print(a_b[:, [0, 2, 4, 6, 8]])

    # this is the value of attribute 'b'
    print(a_b[:, [1, 3, 5, 7, 9]])

if __name__ == "__main__":
    start()