# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from typing import Callable, Dict

from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity

from .config import workflow_settings

keys_in_state = [
    (None, ['is_over_stock', 'is_out_of_stock', 'is_below_rop', 'consumption_hist']),
    ('storage_capacity', ['storage_utilization']),
    ('storage_capacity', [
        'sale_std',
        'sale_hist',
        'pending_order',
        'inventory_in_stock',
        'inventory_in_transit',
        'inventory_estimated',
        'inventory_rop',
    ]),
    ('max_price', ['sku_price', 'sku_cost']),
    (None, ["baseline_action"])
]

# Count the defined state dimension.
list_state_dim = {
    'sale_hist': workflow_settings['sale_hist_len'],
    'consumption_hist': workflow_settings['consumption_hist_len'],
    'pending_order': workflow_settings['pending_order_len'],
}
STATE_DIM = sum(
    list_state_dim[key] if key in list_state_dim else 1
    for _, keys in keys_in_state for key in keys
)

def serialize_state(state: dict) -> np.ndarray:
    result = []

    for norm, fields in keys_in_state:
        for field in fields:
            vals = state[field]
            if not isinstance(vals, list):
                vals = [vals]
            if norm is not None:
                vals = [max(0.0, min(20.0, x / (state[norm] + 0.01))) for x in vals]
            result.extend(vals)

    return np.asarray(result, dtype=np.float32)


class SCAgentStates:
    def __init__(
        self,
        entity_dict: Dict[int, SupplyChainEntity],
        facility_info_dict: Dict[int, FacilityInfo],
        global_sku_id2idx: Dict[int, int],
        sku_number: int,
        max_src_per_facility: int,
        max_price: float,
        settings: dict,
    ) -> None:
        self._facility_info_dict: Dict[int, FacilityInfo] = facility_info_dict
        self._global_sku_id2idx: Dict[int, int] = global_sku_id2idx
        self._sku_number: int = sku_number
        self._max_src_per_facility: int = max_src_per_facility
        self._max_price: float = max_price
        self._settings: dict = settings

        self._atom = self._init_atom()

        self.templates: Dict[int, dict] = {
            entity.id: self._init_entity_state(entity) for entity in entity_dict.values()
        }

    def _init_atom(self) -> Dict[str, Callable]:
        atom = {
            'stock_constraint': (
                lambda f_state: 0 < f_state['inventory_in_stock'] <= (f_state['max_vlt'] + 7) * f_state['sale_mean']
            ),
            'is_replenish_constraint': lambda f_state: f_state['consumption_hist'][-1] > 0,
            'low_profit': lambda f_state: (f_state['sku_price'] - f_state['sku_cost']) * f_state['sale_mean'] <= 1000,
            'low_stock_constraint': (
                lambda f_state: 0 < f_state['inventory_in_stock'] <= (f_state['max_vlt'] + 3) * f_state['sale_mean']
            ),
            'out_of_stock': lambda f_state: 0 < f_state['inventory_in_stock'],
        }

        return atom

    def _init_entity_state(self, entity: SupplyChainEntity) -> dict:
        state: dict = {}
        facility_info: FacilityInfo = self._facility_info_dict[entity.facility_id]

        self._init_global_feature(state)
        self._init_facility_feature(state, entity, facility_info)
        self._init_storage_feature(state, facility_info)
        self._init_bom_feature(state, entity)
        self._init_vlt_feature(state, entity, facility_info)
        self._init_sale_feature(state, entity, facility_info)
        self._init_distribution_feature(state)
        self._init_consumer_feature(state, entity, facility_info)
        self._init_price_feature(state, entity)
        state["baseline_action"] = 0

        return state

    def _init_global_feature(self, state: dict) -> None:
        # state["global_time"] = 0
        return

    def _init_facility_feature(self, state: dict, entity: SupplyChainEntity, facility_info: FacilityInfo) -> None:
        # state["is_positive_balance"] = 0

        # state["facility"] = None
        # state["is_accepted"] = [0] * self._settings["constraint_state_hist_len"]
        # state['constraint_idx'] = [0]
        # state['facility_id'] = [0] * self._sku_number
        # state['sku_info'] = {} if entity.is_facility else entity.skus
        # state['echelon_level'] = 0

        # state['facility_info'] = facility_info.configs

        # if entity.skus is not None:
        #     state['facility_id'][self._global_sku_id2idx[entity.skus.id]] = 1

        # for atom_name in self._atom.keys():
        #     state[atom_name] = list(np.ones(self._settings['constraint_state_hist_len']))

        return

    def _init_storage_feature(self, state: dict, facility_info: FacilityInfo) -> None:
        state['storage_utilization'] = 0
        # state['storage_levels'] = [0] * self._sku_number

        state['storage_capacity'] = 0
        for sub_storage in facility_info.storage_info.config:
            state["storage_capacity"] += sub_storage.capacity

        return

    def _init_bom_feature(self, state: dict, entity: SupplyChainEntity) -> dict:
        # state['bom_inputs'] = [0] * self._sku_number
        # state['bom_outputs'] = [0] * self._sku_number

        # if entity.skus is not None:
        #     state['bom_inputs'][self._global_sku_id2idx[entity.skus.id]] = 1
        #     state['bom_outputs'][self._global_sku_id2idx[entity.skus.id]] = 1
        return

    def _init_vlt_feature(self, state: dict, entity: SupplyChainEntity, facility_info: FacilityInfo) -> None:
        state['max_vlt'] = 0
        # state['vlt'] = [0] * self._max_src_per_facility * self._sku_number
        # # TODO add state['vlt_cost'] for vehicle type selection etc

        if entity.skus is not None:
            product_info = facility_info.products_info[entity.skus.id]

            if product_info.consumer_info is not None:
                state['max_vlt'] = product_info.max_vlt

        #     for i, vlt_info in enumerate(facility_info.upstream_vlt_infos[entity.skus.id]):
        #         vlt_idx = i * self._max_src_per_facility + self._global_sku_id2idx[entity.skus.id]
        #         state['vlt'][vlt_idx] = vlt_info.vlt

        return

    def _init_sale_feature(self, state: dict, entity: SupplyChainEntity, facility_info: FacilityInfo) -> None:
        state['sale_mean'] = 1.0
        state['sale_std'] = 1.0
        # state['sale_gamma'] = 1.0
        # state['total_backlog_demand'] = 0
        state['sale_hist'] = [0] * self._settings['sale_hist_len']
        state['demand_hist'] = [0] * self._settings['sale_hist_len']
        # state['backlog_demand_hist'] = [0] * self._settings['sale_hist_len']
        state['consumption_hist'] = [0] * self._settings['consumption_hist_len']
        state['pending_order'] = [0] * self._settings['pending_order_len']

        state['service_level'] = 0.95

        if entity.skus is not None:
            state['service_level'] = entity.skus.service_level

        #     if facility_info.products_info[entity.skus.id].seller_info is not None:
        #         state['sale_gamma'] = facility_info.skus[entity.skus.id].sale_gamma

        return

    def _init_distribution_feature(self, state: dict) -> None:
        state['distributor_in_transit_orders'] = 0
        state['distributor_in_transit_orders_qty'] = 0
        return

    def _init_consumer_feature(self, state: dict, entity: SupplyChainEntity, facility_info: FacilityInfo) -> None:
        # state['consumer_in_transit_orders'] = [0] * self._sku_number

        state['inventory_in_stock'] = 0
        state['inventory_in_transit'] = 0
        state['inventory_in_distribution'] = 0
        state['inventory_estimated'] = 0

        state['is_over_stock'] = 0
        state['is_out_of_stock'] = 0
        state['is_below_rop'] = 0
        state['inventory_rop'] = 0

        # state['consumer_source_export_mask'] = [0] * self._max_src_per_facility * self._sku_number
        # if entity.skus is not None:
        #     for i, vlt_info in enumerate(facility_info.upstream_vlt_infos[entity.skus.id]):
        #         vlt_idx = i * self._max_src_per_facility + self._global_sku_id2idx[entity.skus.id]
        #         state['consumer_source_export_mask'][vlt_idx] = vlt_info.vlt

        # state['consumer_source_inventory'] = [0] * self._sku_number

        return

    def _init_price_feature(self, state: dict, entity: SupplyChainEntity) -> None:
        state['max_price'] = self._max_price
        state['sku_price'] = 0
        state['sku_cost'] = 0

        if entity.skus is not None:
            # TODO: add property for it. and add it into env metrics
            # state['sku_cost'] = ...
            state['sku_price'] = entity.skus.price

        return
