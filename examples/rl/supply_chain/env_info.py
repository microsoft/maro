# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from collections import defaultdict
from os.path import dirname

import numpy as np

from maro.simulator import Env

sc_path = dirname(__file__)
sys.path.insert(0, sc_path)
from config import env_conf, keys_in_state, q_net_conf


class UnitBaseInfo:
    id: int = None
    node_index: int = None
    config: dict = None
    summary: dict = None

    def __init__(self, unit_summary):
        self.id = unit_summary["id"]
        self.node_index = unit_summary["node_index"]
        self.config = unit_summary.get("config", {})
        self.summary = unit_summary

    def __getitem__(self, key, default=None):
        if key in self.summary:
            return self.summary[key]

        return default


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


atoms = {
    'stock_constraint': stock_constraint,
    'is_replenish_constraint': is_replenish_constraint,
    'low_profit': low_profit,
    'low_stock_constraint': low_stock_constraint,
    'out_of_stock': out_of_stock
}


def get_env_info(env):
    agentid2info = {agent_info.id: agent_info for agent_info in env.agent_idx_list}  # agent (unit id) -> AgentInfo
    num_skus = len(env.summary["node_mapping"]["skus"]) + 1
    facility_levels = {}
    unit2facility = {}
    facility2storage = {}   # facility id -> storage index
    storage_product_num = {}  # facility id -> product id -> number
    storage_product_indexes = defaultdict(dict)  # facility id -> product_id -> index
    facility_product_utilization = {}  # facility id -> storage product utilization
    # use this to quick find relationship between units (consumer, manufacture, seller or product) and product unit.
    # unit id  -> (product unit id, facility id, seller id, consumer id, manufacture id)
    unit2product = {}

    # facility levels
    for facility_id, facility in env.summary["node_mapping"]["facilities"].items():
        facility_levels[facility_id] = {
            "node_index": facility["node_index"],
            "config": facility['configs'],
            "upstreams": facility["upstreams"],
            "skus": facility["skus"]
        }

        units = facility["units"]

        storage = units["storage"]
        if storage is not None:
            facility_levels[facility_id]["storage"] = UnitBaseInfo(
                storage)

            unit2facility[storage["id"]] = facility_id
            facility2storage[facility_id] = storage["node_index"]

            storage_product_num[facility_id] = [0] * num_skus
            facility_product_utilization[facility_id] = 0

            for i, pid in enumerate(storage["product_list"]):
                storage_product_indexes[facility_id][pid] = i
                storage_product_num[facility_id][pid] = 0

        distribution = units["distribution"]

        if distribution is not None:
            facility_levels[facility_id]["distribution"] = UnitBaseInfo(distribution)
            unit2facility[distribution["id"]] = facility_id

        products = units["products"]

        if products:
            for product_id, product in products.items():
                product_info = {
                    "skuproduct": UnitBaseInfo(product)
                }

                unit2facility[product["id"]] = facility_id

                seller = product['seller']

                if seller is not None:
                    product_info["seller"] = UnitBaseInfo(seller)
                    unit2facility[seller["id"]] = facility_id

                consumer = product["consumer"]

                if consumer is not None:
                    product_info["consumer"] = UnitBaseInfo(consumer)
                    unit2facility[consumer["id"]] = facility_id

                manufacture = product["manufacture"]

                if manufacture is not None:
                    product_info["manufacture"] = UnitBaseInfo(manufacture)
                    unit2facility[manufacture["id"]
                                                ] = facility_id

                facility_levels[facility_id][product_id] = product_info

                for unit in (seller, consumer, manufacture, product):
                    if unit is not None:
                        unit2product[unit["id"]] = (
                            product["id"],
                            facility_id,
                            seller["id"] if seller is not None else None,
                            consumer["id"] if consumer is not None else None,
                            manufacture["id"] if manufacture is not None else None
                        )

    # Create initial state structure. We will build the final state with default and const values,
    # then update dynamic part per step
    placeholder_state = {}
    for agent_info in env.agent_idx_list:
        state = {}

        facility = facility_levels[agent_info.facility_id]

        # global features
        state["global_time"] = 0

        # facility features
        state["facility"] = None
        state["facility_type"] = [
            int(i == agent_info.agent_type) for i in range(len(env.summary["node_mapping"]["agent_types"]))
        ]
        state["is_accepted"] = [0] * env.configs.settings["constraint_state_hist_len"]
        state['constraint_idx'] = [0]
        state['facility_id'] = [0] * num_skus
        state['sku_info'] = {} if agent_info.is_facility else agent_info.sku
        state['echelon_level'] = 0

        state['facility_info'] = facility['config']
        state["is_positive_balance"] = 0

        if not agent_info.is_facility:
            state['facility_id'][agent_info.sku.id] = 1

        for atom_name in atoms.keys():
            state[atom_name] = list(
                np.ones(env.configs.settings['constraint_state_hist_len']))

        # storage features
        state['storage_levels'] = [0] * num_skus
        state['storage_capacity'] = facility['storage'].config["capacity"]
        state['storage_utilization'] = 0

        # bom features
        state['bom_inputs'] = [0] * num_skus
        state['bom_outputs'] = [0] * num_skus

        if not agent_info.is_facility:
            state['bom_inputs'][agent_info.sku.id] = 1
            state['bom_outputs'][agent_info.sku.id] = 1

        # vlt features
        sku_list = env.summary["node_mapping"]["skus"]
        current_source_list = []

        if agent_info.sku is not None:
            current_source_list = facility["upstreams"].get(
                agent_info.sku.id, [])

        vlt_len = env.summary["node_mapping"]["max_sources_per_facility"] * num_skus
        state['vlt'] = [0] * vlt_len
        state['max_vlt'] = 0

        if not agent_info.is_facility:
            # only for sku product
            product_info = facility[agent_info.sku.id]

            if "consumer" in product_info and len(current_source_list) > 0:
                state['max_vlt'] = product_info["skuproduct"]["max_vlt"]

                for i, source in enumerate(current_source_list):
                    for j, sku in enumerate(sku_list.values()):
                        # NOTE: different with original code, our config can make sure that source has product we need

                        if sku.id == agent_info.sku.id:
                            state['vlt'][i * len(sku_list) + j +
                                            1] = facility["skus"][sku.id].vlt

        # sale features
        settings = env.configs.settings
        hist_len = settings['sale_hist_len']
        consumption_hist_len = settings['consumption_hist_len']

        state['sale_mean'] = 1.0
        state['sale_std'] = 1.0
        state['sale_gamma'] = 1.0
        state['service_level'] = 0.95
        state['total_backlog_demand'] = 0

        state['sale_hist'] = [0] * hist_len
        state['backlog_demand_hist'] = [0] * hist_len
        state['consumption_hist'] = [0] * consumption_hist_len
        state['pending_order'] = [0] * settings['pending_order_len']

        if not agent_info.is_facility:
            state['service_level'] = agent_info.sku.service_level

            product_info = facility[agent_info.sku.id]

            if "seller" in product_info:
                state['sale_gamma'] = facility["skus"][agent_info.sku.id].sale_gamma

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
                    if sku.id == agent_info.sku.id:
                        state['consumer_source_export_mask'][i * len(sku_list) + j + 1] = \
                            facility_levels[source]["skus"][sku.id].vlt

        # price features
        state['max_price'] = env.summary["node_mapping"]["max_price"]
        state['sku_price'] = 0
        state['sku_cost'] = 0

        if not agent_info.is_facility:
            state['sku_price'] = agent_info.sku.price
            state['sku_cost'] = agent_info.sku.cost

        placeholder_state[agent_info.id] = state

    # state dimension
    state_dim = 0
    first_state = next(iter(placeholder_state.values()))
    for _, state_keys in keys_in_state:
        for key in state_keys:
            val = first_state[key]
            if type(val) == list:
                state_dim += len(val)
            else:
                state_dim += 1

    return {
        "agentid2info": agentid2info,
        "num_skus": len(env.summary["node_mapping"]["skus"]) + 1,
        "facility_levels": facility_levels,
        "unit2facility": unit2facility,
        "facility2storage": facility2storage,
        "storage_product_num": storage_product_num,
        "storage_product_indexes": storage_product_indexes,
        "facility_product_utilization": facility_product_utilization,
        "unit2product": unit2product,
        "placeholder_state": placeholder_state,
        "state_dim": state_dim
    }


q_net_conf["input_dim"] = get_env_info(Env(**env_conf))["state_dim"]
