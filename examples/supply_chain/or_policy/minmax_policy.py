import random as rnd

import numpy as np
import scipy.stats as st

from examples.supply_chain.or_policy.base_policy import ConsumerBaselinePolicy


# parameters: (r, R), calculate according to VLT, demand variances, and service level
# replenish R - S units whenever the current stock is less than r
# S denotes the number of units in stock
class ConsumerMinMaxPolicy(ConsumerBaselinePolicy):

    def __init__(self, name:str, config: dict):
        super(ConsumerMinMaxPolicy, self).__init__(name=name, config=config)

    def compute_action(self, state):
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
        # stop placing orders if no risk of out of stock
        vlt_buffer_days = 10
        vlt = state['vlt'] + vlt_buffer_days
        sale_mean, sale_std = state['sale_mean'], state['sale_std']
        service_level = state['service_level']
        r = (vlt*sale_mean + np.sqrt(vlt)*sale_std*st.norm.ppf(service_level))
        print(booked_inventory, most_needed_product_id, r)
        if booked_inventory[most_needed_product_id] > r:
            return 0
        R = 3*r
        consumer_quantity = int((R - r) / sale_mean)
        return consumer_quantity
