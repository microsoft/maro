# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from .env_helper import STORAGE_INFO, env


def stock_constraint(f_state):
    return 0 < f_state['inventory_in_stock'] <= (f_state['max_vlt'] + 7) * f_state['sale_mean']


def is_replenish_constraint(f_state):
    return f_state['consumption_hist'][-1] > 0


def low_profit(f_state):
    return (f_state['sku_price'] - f_state['sku_cost']) * f_state['sale_mean'] <= 1000


def low_stock_constraint(f_state):
    return 0 < f_state['inventory_in_stock'] <= (f_state['max_vlt'] + 3) * f_state['sale_mean']


def out_of_stock(f_state):
    return 0 < f_state['inventory_in_stock']


workflow_setting = {
    "global_reward_weight_producer": 0.50,
    "global_reward_weight_consumer": 0.50,
    "downsampling_rate": 1,
    "episod_duration": 21,
    "initial_balance": 100000,
    "consumption_hist_len": 4,
    "sale_hist_len": 4,
    "pending_order_len": 4,
    "constraint_state_hist_len": 8,
    "total_echelons": 3,
    "replenishment_discount": 0.9,
    "reward_normalization": 1e7,
    "constraint_violate_reward": -1e6,
    "gamma": 0.99,
    "tail_timesteps": 7,
    "heading_timesteps": 7,
}


atoms = {
    'stock_constraint': stock_constraint,
    'is_replenish_constraint': is_replenish_constraint,
    'low_profit': low_profit,
    'low_stock_constraint': low_stock_constraint,
    'out_of_stock': out_of_stock
}

keys_in_state = [
    (None, ['is_over_stock', 'is_out_of_stock', 'is_below_rop', 'consumption_hist']),
    ('storage_capacity', ['storage_utilization']),
    ('sale_mean', [
        'sale_std',
        'sale_hist',
        'pending_order',
        'inventory_in_stock',
        'inventory_in_transit',
        'inventory_estimated',
        'inventory_rop'
    ]),
    ('max_price', ['sku_price', 'sku_cost'])
]

# Create initial state structure. We will build the final state with default and const values,
# then update dynamic part per step
# print(list(env.summary["node_mapping"].keys()))
num_skus = len(env.summary["node_mapping"]["skus"]) + 1
STATE_TEMPLATE = {}
for entity in env.business_engine.get_entity_list():
    state = {}
    facility = STORAGE_INFO["facility_levels"][entity.facility_id]

    # global features
    state["global_time"] = 0

    # facility features
    state["facility"] = None
    # state["facility_type"] = [
    #     int(i == entity.agent_type) for i in range(len(env.summary["node_mapping"]["agent_types"]))
    # ]
    state["is_accepted"] = [0] * workflow_setting["constraint_state_hist_len"]
    state['constraint_idx'] = [0]
    state['facility_id'] = [0] * num_skus
    state['sku_info'] = {} if entity.is_facility else entity.skus
    state['echelon_level'] = 0

    state['facility_info'] = facility['config']
    state["is_positive_balance"] = 0

    if entity.skus is not None:
        state['facility_id'][entity.skus.id] = 1

    for atom_name in atoms.keys():
        state[atom_name] = list(np.ones(workflow_setting['constraint_state_hist_len']))

    # storage features
    state['storage_levels'] = [0] * num_skus
    state['storage_capacity'] = facility['storage'].config["capacity"]
    state['storage_utilization'] = 0

    # bom features
    state['bom_inputs'] = [0] * num_skus
    state['bom_outputs'] = [0] * num_skus

    if entity.skus is not None:
        state['bom_inputs'][entity.skus.id] = 1
        state['bom_outputs'][entity.skus.id] = 1

    # vlt features
    sku_list = env.summary["node_mapping"]["skus"]
    current_source_list = []

    if entity.skus is not None:
        current_source_list = facility["upstreams"].get(entity.skus.id, [])

    vlt_len = env.summary["node_mapping"]["max_sources_per_facility"] * num_skus
    state['vlt'] = [0] * vlt_len
    state['max_vlt'] = 0

    if entity.skus is not None:
        # only for sku product
        product_info = facility[entity.skus.id]

        if "consumer" in product_info and len(current_source_list) > 0:
            state['max_vlt'] = product_info["skuproduct"]["max_vlt"]

            for i, source in enumerate(current_source_list):
                for j, sku in enumerate(sku_list.values()):
                    # NOTE: different with original code, our config can make sure that source has product we need

                    if sku.id == entity.skus.id:
                        state['vlt'][i * len(sku_list) + j + 1] = facility["skus"][sku.id].vlt

    # sale features
    hist_len = workflow_setting['sale_hist_len']
    consumption_hist_len = workflow_setting['consumption_hist_len']

    state['sale_mean'] = 1.0
    state['sale_std'] = 1.0
    state['sale_gamma'] = 1.0
    state['service_level'] = 0.95
    state['total_backlog_demand'] = 0

    state['sale_hist'] = [0] * hist_len
    state['backlog_demand_hist'] = [0] * hist_len
    state['consumption_hist'] = [0] * consumption_hist_len
    state['pending_order'] = [0] * workflow_setting['pending_order_len']

    if entity.skus is not None:
        state['service_level'] = entity.skus.service_level

        product_info = facility[entity.skus.id]

        if "seller" in product_info:
            state['sale_gamma'] = facility["skus"][entity.skus.id].sale_gamma

    # distribution features
    state['distributor_in_transit_orders'] = 0
    state['distributor_in_transit_orders_qty'] = 0

    # consumer features
    state['consumer_source_export_mask'] = [0] * vlt_len
    state['consumer_source_inventory'] = [0] * num_skus
    state['consumer_in_transit_orders'] = [0] * num_skus

    state['inventory_in_stock'] = 0
    state['inventory_in_transit'] = 0
    state['inventory_in_distribution'] = 0
    state['inventory_estimated'] = 0
    state['inventory_rop'] = 0
    state['is_over_stock'] = 0
    state['is_out_of_stock'] = 0
    state['is_below_rop'] = 0

    if len(current_source_list) > 0:
        for i, source in enumerate(current_source_list):
            for j, sku in enumerate(sku_list.values()):
                if sku.id == entity.skus.id:
                    state['consumer_source_export_mask'][i * len(sku_list) + j + 1] = \
                        STORAGE_INFO["facility_levels"][source]["skus"][sku.id].vlt

    # price features
    state['max_price'] = env.summary["node_mapping"]["max_price"]
    state['sku_price'] = 0
    state['sku_cost'] = 0

    if entity.skus is not None:
        state['sku_price'] = entity.skus.price
        state['sku_cost'] = entity.skus.cost

    STATE_TEMPLATE[entity.id] = state

# state dimension
first_state = next(iter(STATE_TEMPLATE.values()))
STATE_DIM = sum(
    len(first_state[key]) if isinstance(first_state[key], list) else 1
    for _, state_keys in keys_in_state for key in state_keys
)
