# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
A simple script that used to generate random configurations.
"""

import argparse
import os
import random
from typing import Optional

import numpy as np
from flloat.parser.ltlf import LTLfParser
from yaml import safe_dump, safe_load

# Definition of warehouse.
warehouse_def = """
class: "WarehouseFacility"
children:
    storage:
        class: "StorageUnit"
    distribution:
        class: "DistributionUnit"
    products:
        class: "ProductUnit"
        is_template: true
        config:
            agent_type: 4
            consumer:
                class: "ConsumerUnit"
config:
    agent_type: 1
"""

# Definition of supplier.
supplier_def = """
class: "SupplierFacility"
children:
    storage:
        class: "StorageUnit"
    distribution:
        class: "DistributionUnit"
    products:
        class: "ProductUnit"
        is_template: true
        config:
            agent_type: 3
            consumer:
                class: "ConsumerUnit"
            manufacture:
                class: "ManufactureUnit"
config:
    agent_type: 0
"""

# Definition of retailer.
retailer_def = """
class: "RetailerFacility"
children:
    storage:
        class: "StorageUnit"
    products:
        class: "StoreProductUnit"
        is_template: true
        config:
            agent_type: 5
            consumer:
                class: "ConsumerUnit"
            seller:
                class: "SellerUnit"
                config:
                    sale_hist_len: 4
config:
    agent_type: 2
"""

# Template to generate a supplier facility.
# Properties to change:
# . name
# . skus
# . vehicles
# . config (optional)
supplier_template = """
name: "Supplier_001"
definition_ref: "SupplierFacility"
skus: {}
children:
    storage:
        config:
            capacity: 10000
            unit_storage_cost: 1
    distribution:
        children:
            vehicles: []
        config:
            unit_price: 1
config: {}
"""

# Template to generate warehouse facility.
# Property to change:
# . name
# . skus
# . vehicles
# . config (optional)
warehouse_template = """
name: "Warehouse_001"
definition_ref: "WarehouseFacility"
skus: {}
children:
    storage:
        config:
            capacity: 10000
            unit_storage_cost: 1
    distribution:
        children:
            vehicles: []
        config:
            unit_price: 1
config: {}
"""

# Template to generate retailer.
# Property to change:
# . name
# . skus
# . config (optional)
retailer_template = """
name: "Retailer_001"
definition_ref: "RetailerFacility"
skus: {}
children:
    storage:
        config:
            capacity: 10000
            unit_storage_cost: 1
config: {}
"""


def generate_config(
    sku_num: int,
    supplier_num: int,
    warehouse_num: int,
    retailer_num: int,
    grid_width: int,
    grid_height: int,
    output_path: Optional[str] = None,
):
    constraints = [
        "G(stock_constraint)",
        "G(is_replenish_constraint -> ((X!is_replenish_constraint)&(XX!is_replenish_constraint)))",
        "G(low_profit -> low_stock_constraint)",
    ]

    # constraints = ['G(is_replenish_constraint -> ((X!is_replenish_constraint)&(XX!is_replenish_constraint)))']

    def construct_formula(constraint):
        parser = LTLfParser()
        formula = parser(constraint)
        return formula

    constraint_formulas = {constraint: construct_formula(constraint) for constraint in constraints}
    constraint_automata = {
        constraint: constraint_formulas[constraint].to_automaton().determinize() for constraint in constraints
    }

    max_constraint_states = int(np.max([len(a.states) for a in constraint_automata.values()]))

    # Base configuration of vehicle used in all facility.
    vehicle_conf = {
        "class": "VehicleUnit",
        "config": {
            "patient": 100,
            "unit_transport_cost": 1,
        },
    }

    # Save the vehicle definition in the config, so later distribution will reference to it.
    config = {
        "normal_vehicle": vehicle_conf,
        "facility_definitions": {},
        "settings": {
            "global_reward_weight_producer": 0.50,
            "global_reward_weight_consumer": 0.50,
            "downsampling_rate": 1,
            "episod_duration": 21,
            "initial_balance": 100000,
            "consumption_hist_len": 4,
            "sale_hist_len": 4,
            "pending_order_len": 4,
            "constraint_state_hist_len": max_constraint_states,
            "total_echelons": 3,
            "replenishment_discount": 0.9,
            "reward_normalization": 1e7,
            "constraint_violate_reward": -1e6,
            "gamma": 0.99,
            "tail_timesteps": 7,
            "heading_timesteps": 7,
        },
    }

    # Add the facility definitions.
    config["facility_definitions"]["SupplierFacility"] = safe_load(supplier_def)
    config["facility_definitions"]["WarehouseFacility"] = safe_load(warehouse_def)
    config["facility_definitions"]["RetailerFacility"] = safe_load(retailer_def)

    # Generate settings first.
    world_conf = {}

    sku_names = [f"SKU{i}" for i in range(sku_num)]

    sku_list = []
    for sku_index, sku_name in enumerate(sku_names):
        sku_list.append(
            {
                "id": sku_index,
                "name": sku_name,
            },
        )

    # Add the sku list to the world configuration.
    world_conf["skus"] = sku_list

    # Generate sku information.
    sku_cost = {f"SKU{i}": random.randint(10, 500) for i in range(sku_num)}
    sku_product_cost = {f"SKU{i}": int(sku_cost[f"SKU{i}"] * 0.9) for i in range(sku_num)}
    sku_price = {f"SKU{i}": int(sku_cost[f"SKU{i}"] * (1 + random.randint(10, 100) / 100)) for i in range(sku_num)}
    sku_gamma = {f"SKU{i}": random.randint(5, 100) for i in range(sku_num)}
    total_gamma = sum(list(sku_gamma.values()))
    sku_vlt = {f"SKU{i}": random.randint(1, 3) for i in range(sku_num)}

    # Generate suppliers.
    supplier_facilities = []

    for i in range(supplier_num):
        facility = safe_load(supplier_template)

        facility["name"] = f"SUPPLIER{i}"
        facility["children"]["storage"]["config"]["capacity"] = total_gamma * 100

        for _ in range(10 * sku_num):
            # this will save as a reference in the final yaml file
            facility["children"]["distribution"]["children"]["vehicles"].append(vehicle_conf)

        # Facility config.
        facility["config"] = {}
        facility["config"]["unit_order_cost"] = 200
        facility["config"]["delay_order_penalty"] = 1000

        # Sku list of this facility.
        sku_list = {}

        for j in range(sku_num):
            sku_name = f"SKU{j}"
            sku_list[sku_name] = {
                "price": sku_cost[sku_name],
                "cost": sku_product_cost[sku_name],
                "service_level": 0.95,
                "vlt": 3,
                "init_stock": int(sku_gamma[sku_name] * 50),
                # Why this configuration, as manufacture is controlled by action?
                "production_rate": int(sku_gamma[sku_name] * 50),
                # For this script, all sku is a production that produced by suppliers, no bom.
                "type": "production",
                "product_unit_cost": 1,
            }

        facility["skus"] = sku_list

        supplier_facilities.append(facility)

    # Warehouses.
    warehouse_list = []
    for i in range(warehouse_num):
        facility = safe_load(warehouse_template)

        facility["name"] = f"WAREHOUSE{i}"
        facility["children"]["storage"]["config"]["capacity"] = total_gamma * 100

        for _ in range(10 * sku_num):
            facility["children"]["distribution"]["children"]["vehicles"].append(vehicle_conf)

        facility["config"] = {}
        facility["config"]["unit_order_cost"] = 500
        facility["config"]["delay_order_penalty"] = 1000

        sku_list = {}

        for j in range(sku_num):
            sku_name = f"SKU{j}"
            sku_list[sku_name] = {
                "price": sku_cost[sku_name],
                "cost": sku_cost[sku_name],
                "vlt": sku_vlt[sku_name],
                "init_stock": int(sku_gamma[sku_name] * 20),
                "service_level": 0.96,
            }

        facility["skus"] = sku_list

        warehouse_list.append(facility)

    sku_constraints = {}
    for i in range(sku_num):
        if random.random() <= 0.5:
            continue
        sku_constraints[f"SKU{i}"] = constraints[random.randint(0, len(constraints) - 1)]

    # Retailers.
    retailer_list = []
    for i in range(retailer_num):
        facility = safe_load(retailer_template)

        facility["name"] = f"STORE{i}"
        facility["children"]["storage"]["config"]["capacity"] = total_gamma * 20

        facility["config"] = {}
        facility["config"]["unit_order_cost"] = 500

        sku_list = {}

        for j in range(sku_num):
            sku_name = f"SKU{j}"
            sku_list[sku_name] = {
                "price": sku_price[sku_name],
                "service_level": 0.95,
                "cost": sku_cost[sku_name],
                "init_stock": sku_gamma[sku_name] * (sku_vlt[sku_name] + random.randint(1, 5)),
                "sale_gamma": sku_gamma[sku_name],
                "max_stock": 1000,
                "constraint": sku_constraints.get(sku_name, None),
            }

        facility["skus"] = sku_list

        retailer_list.append(facility)

    world_conf["facilities"] = supplier_facilities + warehouse_list + retailer_list

    # According to original code, the upstream relationship is like following:
    # supplier <- warehouse <- retailer
    # as current configuration supplier and warehouse contain all the sku, so we can just random pick.
    world_conf["topology"] = {}

    # Random pick upstreams for retailers from warehouses.
    for store in retailer_list:
        store_upstream = {}

        for i in range(sku_num):
            sku_name = f"SKU{i}"
            store_upstream[sku_name] = [warehouse_list[random.randint(0, warehouse_num - 1)]["name"]]

        world_conf["topology"][store["name"]] = store_upstream

    # Random pick upstreams for warehouses from suppliers.
    for warehouse in warehouse_list:
        warehouse_upstream = {}

        for i in range(sku_num):
            sku_name = f"SKU{i}"
            warehouse_upstream[sku_name] = [supplier_facilities[random.randint(0, supplier_num) - 1]["name"]]

        world_conf["topology"][warehouse["name"]] = warehouse_upstream

    # Grid settings.
    world_conf["grid"] = {}
    world_conf["grid"]["size"] = [grid_width, grid_height]

    # Random pick location.
    available_cells = [(x, y) for x in range(grid_width) for y in range(grid_height)]

    world_conf["grid"]["facilities"] = {}
    for facility in world_conf["facilities"]:
        cell = random.randint(0, len(available_cells) - 1)

        world_conf["grid"]["facilities"][facility["name"]] = available_cells[cell]

        del available_cells[cell]

    config["world"] = world_conf

    if output_path is None:
        output_path = "."

    with open(os.path.join(output_path, "config.yml"), "wt+") as fp:
        safe_dump(config, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sku_num", type=int, default=random.randint(4, 5))
    parser.add_argument("--supplier_num", type=int, default=1)
    parser.add_argument("--warehouse_num", type=int, default=1)
    parser.add_argument("--retailer_num", type=int, default=1)
    parser.add_argument("--grid_width", type=int, default=20)
    parser.add_argument("--grid_height", type=int, default=20)
    parser.add_argument("--output_path", type=str, default=".")

    arg = parser.parse_args()

    generate_config(
        arg.sku_num,
        arg.supplier_num,
        arg.warehouse_num,
        arg.retailer_num,
        arg.grid_width,
        arg.grid_height,
        arg.output_path,
    )
