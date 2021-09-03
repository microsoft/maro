# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import sys

import numpy as np
import scipy.stats as st

from maro.rl.exploration import EpsilonGreedyExploration, LinearExplorationScheduler
from maro.rl.modeling import FullyConnected, OptimOption, DiscreteQNet
from maro.rl.policy import DQN, AbsPolicy, NullPolicy

cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
from config import NUM_CONSUMER_ACTIONS, NUM_RL_POLICIES, dqn_conf, exploration_conf, q_net_conf


######################################## DQN #################################################
class QNet(DiscreteQNet):
    def __init__(self, component, optim_option=None, device=None):
        super().__init__(component, optim_option=optim_option, device=device)

    @property
    def input_dim(self):
        return self.component.input_dim

    def forward(self, states):
        return self.component(states)


def get_dqn(name: str):
    qnet = QNet(FullyConnected(**q_net_conf["network"]), optim_option=OptimOption(**q_net_conf["optimization"]))
    exploration = EpsilonGreedyExploration()
    exploration.register_schedule(
        scheduler_cls=LinearExplorationScheduler,
        param_name="epsilon",
        **exploration_conf
    )
    return DQN(name, qnet, exploration=exploration, **dqn_conf)


######################################## Rule-based / OR #################################################

class ProducerBaselinePolicy(AbsPolicy):
    def choose_action(self, state):
        return state.get('product_rate', 500)


class ConsumerBaselinePolicy(AbsPolicy):
    def __init__(self, name, num_actions: int):
        super().__init__(name)
        self.num_actions = num_actions

    def choose_action(self, state):
        if state['is_facility']:
            return 0
        # consumer_source_inventory
        available_inventory = np.array(state['storage_levels'])
        inflight_orders = np.array(state['consumer_in_transit_orders'])
        booked_inventory = available_inventory + inflight_orders

        # stop placing orders when the facility runs out of capacity
        if np.sum(booked_inventory) > state['storage_capacity']:
            return 0

        most_needed_product_id = state['product_idx']
        sale_mean, sale_std = state['sale_mean'], state['sale_std']
        service_level = state['service_level']
        vlt_buffer_days = 7
        vlt = vlt_buffer_days + state['vlt']
        if booked_inventory[most_needed_product_id] > vlt*sale_mean + np.sqrt(vlt)*sale_std*st.norm.ppf(service_level):
            return 0
        consumer_action_space_size = self.num_actions
        consumer_quantity = random.randint(0, consumer_action_space_size-1)
        return consumer_quantity


# Q = \sqrt{2DK/h}
# Q - optimal order quantity
# D - annual demand quantity
# K - fixed cost per order, setup cost (not per unit, typically cost of ordering and shipping and handling. This is not the cost of goods)
# h - annual holding cost per unit,
#     also known as carrying cost or storage cost (capital cost, warehouse space,
#     refrigeration, insurance, etc. usually not related to the unit production cost)
class ConsumerEOQPolicy(AbsPolicy):
    def _get_consumer_quantity(self, state):
        order_cost = state['order_cost']
        holding_cost = state['unit_storage_cost']
        sale_gamma = state['sale_mean']
        consumer_quantity = int(np.sqrt(2*sale_gamma*order_cost / holding_cost) / sale_gamma)
        return consumer_quantity

    def choose_action(self, state):
        if state['is_facility']:
            return 0
        # consumer_source_inventory
        available_inventory = np.array(state['storage_levels'])
        inflight_orders = np.array(state['consumer_in_transit_orders'])
        booked_inventory = available_inventory + inflight_orders

        # stop placing orders when the facilty runs out of capacity
        if np.sum(booked_inventory) > state['storage_capacity']:
            return 0

        most_needed_product_id = state['product_idx']
        vlt_buffer_days = 7
        vlt = vlt_buffer_days + state['vlt']
        sale_mean, sale_std = state['sale_mean'], state['sale_std']
        service_level = state['service_level']

        # whether replenishment point is reached
        if booked_inventory[most_needed_product_id] > vlt*sale_mean + np.sqrt(vlt)*sale_std*st.norm.ppf(service_level):
            return 0
        consumer_quantity = self._get_consumer_quantity(state)
        return consumer_quantity


# parameters: (r, R), calculate according to VLT, demand variances, and service level
# replenish R - S units whenever the current stock is less than r
# S denotes the number of units in stock
class ConsumerMinMaxPolicy(AbsPolicy):
    def choose_action(self, state):
        if state['is_facility']:
            return 0
        # consumer_source_inventory
        available_inventory = np.array(state['storage_levels'])
        inflight_orders = np.array(state['consumer_in_transit_orders'])
        booked_inventory = available_inventory + inflight_orders

        # stop placing orders when the facility runs out of capacity
        if np.sum(booked_inventory) > state['storage_capacity']:
            return 0

        most_needed_product_id = state['product_idx']
        # stop placing orders if no risk of out of stock
        vlt_buffer_days = 10
        vlt = state['vlt'] + vlt_buffer_days
        sale_mean, sale_std = state['sale_mean'], state['sale_std']
        service_level = state['service_level']
        r = (vlt*sale_mean + np.sqrt(vlt)*sale_std*st.norm.ppf(service_level))
        # print(booked_inventory, most_needed_product_id, r)
        if booked_inventory[most_needed_product_id] > r:
            return 0
        R = 3*r
        consumer_quantity = int((R - r) / sale_mean)
        return consumer_quantity


# def get_consumer_baseline_policy():
#     return ConsumerBaselinePolicy(NUM_CONSUMER_ACTIONS)

# def get_consumer_eoq_policy():
#     return ConsumerEOQPolicy()

or_policy_func_dict = {
    # "consumer": lambda: ConsumerMinMaxPolicy(),
    "producer": lambda name: ProducerBaselinePolicy(name),
    "facility": lambda name: NullPolicy(name),
    "product": lambda name: NullPolicy(name),
    "productstore": lambda name: NullPolicy(name)
}
policy_func_dict = {f"consumer-{i}": get_dqn for i in range(NUM_RL_POLICIES)}
