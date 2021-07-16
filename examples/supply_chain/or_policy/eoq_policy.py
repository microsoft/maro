# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random as rnd

import numpy as np
import scipy.stats as st

from examples.supply_chain.or_policy.base_policy import ConsumerBaselinePolicy


# Q = \sqrt{2DK/h}
# Q - optimal order quantity
# D - annual demand quantity
# K - fixed cost per order, setup cost (not per unit, typically cost of ordering and shipping and handling. This is not the cost of goods)
# h - annual holding cost per unit,
#     also known as carrying cost or storage cost (capital cost, warehouse space,
#     refrigeration, insurance, etc. usually not related to the unit production cost)
class ConsumerEOQPolicy(ConsumerBaselinePolicy):

    def __init__(self, name: str, config: dict):
        super(ConsumerEOQPolicy, self).__init__(name=name, config=config)

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
