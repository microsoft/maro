import numpy as np
from maro.rl.policy import AbsFixedPolicy
import random
import scipy.stats as st


class ProducerBaselinePolicy(AbsFixedPolicy):
    
    def __init__(self, config):
        self.config = config
        # self.num_actions = config["model"]["network"]["output_dim"]

    def choose_action(self, state):
        return state.get('product_rate', 500)


class ConsumerBaselinePolicy(AbsFixedPolicy):

    def __init__(self, config):
        self.config = config
        self.num_actions = config["model"]["network"]["output_dim"]

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
        sale_mean, sale_std = state['sale_mean'], state['sale_std']
        service_level = state['service_level']
        vlt_buffer_days = 7
        vlt = vlt_buffer_days + state['vlt']
        if booked_inventory[most_needed_product_id] > vlt*sale_mean + np.sqrt(vlt)*sale_std*st.norm.ppf(service_level):
            return 0
        consumer_action_space_size = self.num_actions
        consumer_quantity = random.randint(0, consumer_action_space_size-1)
        return consumer_quantity
