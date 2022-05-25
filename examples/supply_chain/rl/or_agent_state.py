# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List, Optional

import numpy as np
import scipy.stats as st

from maro.simulator.scenarios.supply_chain.facilities import FacilityBase, FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity, VendorLeadingTimeInfo


class ScOrAgentStates:
    def __init__(
        self,
        entity_dict: Dict[int, SupplyChainEntity],
        facility_info_dict: Dict[int, FacilityInfo],
        global_sku_id2idx: Dict[int, int],
    ) -> None:
        self._entity_dict: Dict[int, SupplyChainEntity] = entity_dict
        self._facility_info_dict: Dict[int, FacilityInfo] = facility_info_dict
        self._global_sku_id2idx: Dict[int, int] = global_sku_id2idx

        self._storage_capacity_dict: Optional[Dict[int, Dict[int, int]]] = None

        self._templates: Dict[int, dict] = {}

    def _init_entity_state(
        self,
        entity: SupplyChainEntity,
        chosen_vlt_info: Optional[VendorLeadingTimeInfo],
        fixed_vlt: bool,
    ) -> dict:
        facility_info = self._facility_info_dict[entity.facility_id]
        storage_index = facility_info.storage_info.node_index

        max_vlt = facility_info.products_info[entity.skus.id].max_vlt
        if fixed_vlt and chosen_vlt_info is not None:
            max_vlt = chosen_vlt_info.vlt

        state: dict = {
            "demand_mean": 0,
            "demand_std": 0,
            "unit_storage_cost": entity.skus.unit_storage_cost,
            "unit_order_cost": entity.skus.unit_order_cost,
            "storage_capacity": self._storage_capacity_dict[storage_index][entity.skus.id],
            "storage_utilization": 0,
            "storage_in_transition_quantity": 0,
            "product_level": 0,
            "in_transition_quantity": 0,
            "to_distribute_quantity": 0,
            "cur_vlt": chosen_vlt_info.vlt + 1 if chosen_vlt_info else 0,
            "max_vlt": max_vlt + 1,
            "service_level_ppf": st.norm.ppf(entity.skus.service_level),
        }

        return state

    def update_entity_state(
        self,
        entity_id: int,
        storage_capacity_dict: Optional[Dict[int, Dict[int, int]]],
        product_metrics: Optional[dict],
        product_levels: List[int],
        in_transit_quantity: List[int],
        to_distribute_quantity: List[int],
        history_demand: np.ndarray,
        history_price: np.ndarray,
        chosen_vlt_info: Optional[VendorLeadingTimeInfo],
        fixed_vlt: bool,
    ) -> dict:
        entity: SupplyChainEntity = self._entity_dict[entity_id]

        if issubclass(entity.class_type, FacilityBase):
            return {}

        if entity_id not in self._templates:
            if self._storage_capacity_dict is None:
                self._storage_capacity_dict = storage_capacity_dict

            self._templates[entity_id] = self._init_entity_state(entity, chosen_vlt_info, fixed_vlt)

        state: dict = self._templates[entity_id]

        state["demand_mean"] = product_metrics["demand_mean"]
        state["demand_std"] = product_metrics["demand_std"]

        state["storage_utilization"] = sum(product_levels)
        state["storage_in_transition_quantity"] = sum(in_transit_quantity)

        state["product_level"] = product_levels[self._global_sku_id2idx[entity.skus.id]]
        state["in_transition_quantity"] = in_transit_quantity[self._global_sku_id2idx[entity.skus.id]]
        state["to_distribute_quantity"] = to_distribute_quantity[self._global_sku_id2idx[entity.skus.id]]

        state["cur_vlt"] = chosen_vlt_info.vlt + 1 if chosen_vlt_info else 0
        state["entity_id"] = entity_id

        product_info = self._facility_info_dict[entity.facility_id].products_info[entity.skus.id]
        if product_info.seller_info is not None:
            seller_index = product_info.seller_info.node_index
            product_index = product_info.node_index
            state["history_demand"] = history_demand[:, seller_index]
            state["history_price"] = history_price[:, product_index]
        return state
