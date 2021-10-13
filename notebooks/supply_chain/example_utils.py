from typing import List

import numpy as np

from maro.simulator import Env


def get_facility_name2id_mapping(env: Env) -> dict:
    facility_name2id = {}
    for facility_id, facility_info in env.summary["node_mapping"]["facilities"].items():
        facility_name = facility_info["name"]
        facility_name2id[facility_name] = facility_id
    return facility_name2id


def get_unit_name2id_mapping(env: Env) -> dict:
    unit_name2id = {}
    for facility_id, facility_info in env.summary["node_mapping"]["facilities"].items():
        facility_name = facility_info["name"]
        for unit_name, unit_info in facility_info['units'].items():
            if unit_info is None:
                continue
            if unit_name == "products":
                for sku_id, sku_info in unit_info.items():
                    unit_name2id[f"{facility_name}__{unit_name}{sku_id}"] = sku_info["id"]
                    for prod_name in ('consumer', 'manufacture', 'seller'):
                        if sku_info[prod_name] is not None:
                            unit_name2id[f"{facility_name}__{prod_name}{sku_id}"] = sku_info[prod_name]["id"]
            else:
                unit_name2id[f"{facility_name}__{unit_name}"] = unit_info["id"]
    return unit_name2id


def get_storage_status(env: Env, facility_info: dict) -> List:
    storage_nodes = env.snapshot_list["storage"]
    storage_index = facility_info["units"]["storage"]["node_index"]
    storage_states = storage_nodes[env.frame_index:storage_index:("capacity", "remaining_space")]\
        .flatten().astype(np.int)

    sku_list = storage_nodes[env.frame_index:storage_index:["product_list"]].flatten().astype(np.int)
    sku_num = storage_nodes[env.frame_index:storage_index:["product_number"]].flatten().astype(np.int)

    return [storage_states[0], storage_states[1], dict(zip(sku_list, sku_num))]  # (capacity, remain, sku_detail)


def get_manufacture_status(env: Env, facility_info: dict, sku_id: int) -> List:
    manufacture_nodes = env.snapshot_list["manufacture"]
    manufacture_features = ("manufacturing_number", "product_unit_cost")
    manufacture_unit = facility_info["units"]["products"][sku_id]["manufacture"]
    if manufacture_unit is None:  # This facility does not have a manufacture unit for this SKU
        return [None, None]
    manufacture_index = manufacture_unit["node_index"]
    manufacture_states = manufacture_nodes[env.frame_index:manufacture_index:manufacture_features].flatten().astype(
        np.int)
    return manufacture_states  # (manufacturing_number, unit_cost)


def show_storage_status(env: Env) -> None:
    buff = {}
    sku_id_set = set([])
    for facility_id, facility_info in env.summary["node_mapping"]["facilities"].items():
        capacity, remaining, sku_detail = get_storage_status(env, facility_info)
        buff[(facility_id, facility_info["name"])] = (capacity, remaining, sku_detail)
        sku_id_set |= set(sku_detail.keys())
    sku_id_list = sorted(list(sku_id_set))

    print("  Storage summary:")
    print("    {}{} / {}".format("[Facility]".ljust(16), "[remain]".rjust(8), "[capacity]".rjust(10)))
    for facility_id, facility_info in env.summary["node_mapping"]["facilities"].items():
        capacity, remaining, _ = buff[(facility_id, facility_info["name"])]
        print("    {}{:8d} / {:10d}".format(facility_info["name"].ljust(16), remaining, capacity))

    print("  Storage detail:")
    print("    " + "[Facility]".ljust(16) + "".join([f"[sku{sku_id}]".rjust(7) for sku_id in sku_id_list]))
    for facility_id, facility_info in env.summary["node_mapping"]["facilities"].items():
        _, _, sku_detail = buff[(facility_id, facility_info["name"])]
        print("    " + facility_info["name"].ljust(16) + "".join(
            [f"{sku_detail.get(sku_id, 0)}".rjust(7) for sku_id in sku_id_list]))


def show_manufacture_status(env: Env) -> None:
    print("  Manufacture status:")
    print("    " + "[Facility]".ljust(16))
    for facility_id, facility_info in env.summary["node_mapping"]["facilities"].items():
        for sku_id in facility_info["units"]["products"]:
            manufacturing_number, unit_cost = get_manufacture_status(env, facility_info, sku_id)
            if manufacturing_number is None:
                continue
            facility_name = facility_info["name"]
            print(
                f"    {facility_name.ljust(16)}sku_id = {sku_id}, manufacturing = {manufacturing_number}, "
                f"unit cost = {unit_cost}")


def show_status(env: Env) -> None:
    print(f"*** Env status after tick {env.tick} ***")
    show_storage_status(env)
    show_manufacture_status(env)
