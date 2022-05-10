# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import scipy.stats as st
from typing import Callable, Dict, List

from maro.simulator.scenarios.supply_chain import ConsumerUnit, ProductUnit
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity

from .config import (
    OR_NUM_CONSUMER_ACTIONS,
    workflow_settings,
    IDX_DISTRIBUTION_PENDING_PRODUCT_QUANTITY, IDX_DISTRIBUTION_PENDING_ORDER_NUMBER,
    IDX_SELLER_TOTAL_DEMAND, IDX_SELLER_SOLD, IDX_SELLER_DEMAND,
    IDX_CONSUMER_ORDER_BASE_COST, IDX_CONSUMER_LATEST_CONSUMPTIONS,
)

keys_in_state = [
    (None, ['is_over_stock', 'is_out_of_stock', 'is_below_rop', 'consumption_hist']),
    ('storage_capacity', ['storage_utilization']),
    ('storage_capacity', [
        'sale_mean',
        'sale_std',
        'sale_hist',
        'demand_hist',
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
    'demand_hist': workflow_settings['sale_hist_len'],
    'consumption_hist': workflow_settings['consumption_hist_len'],
    'pending_order': workflow_settings['pending_order_len'],
    'baseline_action': OR_NUM_CONSUMER_ACTIONS
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
                vals = [max(0.0, min(10.0, x / (state[norm] + 0.01))) for x in vals]
            result.extend(vals)

    return np.asarray(result, dtype=np.float32)


class ScRlAgentStates:
    def __init__(
        self,
        entity_dict: Dict[int, SupplyChainEntity],
        facility_info_dict: Dict[int, FacilityInfo],
        global_sku_id2idx: Dict[int, int],
        sku_number: int,
        max_src_per_facility: int,
        max_price_dict: Dict[int, float],
        settings: dict,
    ) -> None:
        self._entity_dict: Dict[int, SupplyChainEntity] = entity_dict
        self._facility_info_dict: Dict[int, FacilityInfo] = facility_info_dict
        self._global_sku_id2idx: Dict[int, int] = global_sku_id2idx
        self._sku_number: int = sku_number
        self._max_src_per_facility: int = max_src_per_facility
        self._max_price_dict: Dict[int, float] = max_price_dict
        self._settings: dict = settings

        self._atom = self._init_atom()

        # Key: service level; Value: the cached return value of Percentage Point Function.
        self._service_index_ppf_cache: Dict[float, float] = {}

        self._templates: Dict[int, dict] = {}

    @staticmethod
    def _init_atom() -> Dict[str, Callable]:
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
        state["baseline_action"] = [0] * OR_NUM_CONSUMER_ACTIONS

        return state

    def update_entity_state(
        self,
        entity_id: int,
        tick: int,
        cur_metrics: dict,
        cur_distribution_states: np.ndarray,
        cur_seller_hist_states: np.ndarray,
        cur_consumer_hist_states: np.ndarray,
        accumulated_balance: float,
        storage_product_quantity: Dict[int, List[int]],
        facility_product_utilization: Dict[int, int],
        facility_in_transit_orders: Dict[int, List[int]],
    ) -> dict:
        """Update the state dict of the given entity_id in the given tick.

        Args:
            entity_id (int): The id of the target entity unit.
            tick (int): The target environment tick.
            cur_metrics (dict): The environment metrics of the given tick. It is an attribution of the business engine.
            cur_distribution_states (np.ndarray): The distribution attributes of current tick, extracted from the
                snapshot list.
            cur_seller_hist_states (np.ndarray): The seller attributes of the pre-defined time window, extracted from the
                snapshot list.
            cur_consumer_hist_states (np.ndarray): The consumer attributes of the pre-defined time window, extracted from the
                snapshot list.
            accumulated_balance (float): The accumulated balance of the given entity in the given tick.
            storage_product_quantity (Dict[int, List[int]]): The current product quantity in the facility's storage. The
                key is the id of the facility the entity belongs to, the value is the product quantity list which is
                indexed by the global_sku_id2idx.
            facility_product_utilization (Dict[int, int]): The current total product quantity in corresponding facility.
                The key is the id of the facility the entity belongs to, the value is the total product quantity.
            facility_in_transit_orders (Dict[int, List[int]]): The current in-transition product quantity. The key is
                the id of the facility the entity belongs to, the value is the in-transition product quantity list which
                is indexed by the global_sku_id2idx.

        Returns:
            dict: The state dict of the given entity in the given tick.
        """
        entity: SupplyChainEntity = self._entity_dict[entity_id]

        if entity_id not in self._templates:
            self._templates[entity_id] = self._init_entity_state(entity)
        state: dict = self._templates[entity_id]

        self._update_global_features(state, tick)
        self._update_facility_features(state, accumulated_balance)
        self._update_storage_features(state, entity, storage_product_quantity, facility_product_utilization)
        self._update_sale_features(state, entity, cur_metrics, cur_seller_hist_states, cur_consumer_hist_states)
        self._update_distribution_features(state, entity, cur_distribution_states)
        self._update_consumer_features(state, entity, cur_metrics, storage_product_quantity, facility_in_transit_orders)

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
        # state['sku_info'] = {} if issubclass(entity.class_type, FacilityBase) else entity.skus
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
        for sub_storage in facility_info.storage_info.config.values():
            state["storage_capacity"] += sub_storage.capacity

        return

    def _init_distribution_feature(self, state: dict) -> None:
        state['distributor_in_transit_orders'] = 0
        state['distributor_in_transit_orders_qty'] = 0
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

        if entity.skus is not None:
            product_info = facility_info.products_info[entity.skus.id]

            if product_info.consumer_info is not None:
                state['max_vlt'] = product_info.max_vlt

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

        # state['consumer_source_inventory'] = [0] * self._sku_number

        return

    def _init_price_feature(self, state: dict, entity: SupplyChainEntity) -> None:
        state['max_price'] = self._max_price_dict[entity.facility_id]
        state['sku_price'] = 0
        state['sku_cost'] = 0

        if entity.skus is not None:
            # TODO: add property for it. and add it into env metrics
            # state['sku_cost'] = ...
            state['sku_price'] = entity.skus.price

        return

    def _update_global_features(self, state: dict, tick: int) -> None:
        # state["global_time"] = tick
        return

    def _update_facility_features(self, state: dict, accumulated_balance: float) -> None:
        # state['is_positive_balance'] = 1 if accumulated_balance > 0 else 0
        return

    def _update_storage_features(
        self,
        state: dict,
        entity: SupplyChainEntity,
        storage_product_quantity: Dict[int, List[int]],
        facility_product_utilization: Dict[int, int],
    ) -> None:
        # state['storage_levels'] = storage_product_quantity[entity.facility_id]
        state['storage_utilization'] = facility_product_utilization[entity.facility_id]
        return

    def _update_distribution_features(
        self,
        state: dict,
        entity: SupplyChainEntity,
        cur_distribution_states: np.ndarray,
    ) -> None:
        distribution_info = self._facility_info_dict[entity.facility_id].distribution_info
        if distribution_info is not None:
            dist_states = cur_distribution_states[distribution_info.node_index]
            state['distributor_in_transit_orders_qty'] = dist_states[IDX_DISTRIBUTION_PENDING_PRODUCT_QUANTITY]
            state['distributor_in_transit_orders'] = dist_states[IDX_DISTRIBUTION_PENDING_ORDER_NUMBER]
        return

    def _update_sale_features(
        self,
        state: dict,
        entity: SupplyChainEntity,
        cur_metrics: dict,
        cur_seller_hist_states: np.ndarray,
        cur_consumer_hist_states: np.ndarray,
    ) -> None:
        if not issubclass(entity.class_type, (ConsumerUnit, ProductUnit)):
            return

        # Get product unit id for current agent.
        product_unit_id = entity.id if issubclass(entity.class_type, ProductUnit) else entity.parent_id

        state['sale_mean'] = cur_metrics["products"][product_unit_id]["sale_mean"]
        state['sale_std'] = cur_metrics["products"][product_unit_id]["sale_std"]

        product_info = self._facility_info_dict[entity.facility_id].products_info[entity.skus.id]

        if product_info.seller_info is not None:
            seller_states = cur_seller_hist_states[:, product_info.seller_info.node_index, :]

            # For total demand, we need latest one.
            # state['total_backlog_demand'] = seller_states[:, IDX_SELLER_TOTAL_DEMAND][-1][0]
            state['sale_hist'] = list(seller_states[:, IDX_SELLER_SOLD].flatten())
            state['demand_hist'] = list(seller_states[:, IDX_SELLER_DEMAND].flatten())
            # state['backlog_demand_hist'] = list(seller_states[:, IDX_SELLER_DEMAND])
            # print(state['sale_hist'], state['demand_hist'])

        else:
            # state['sale_gamma'] = state['sale_mean']
            pass

        if product_info.consumer_info is not None:
            consumer_states = cur_consumer_hist_states[:, product_info.consumer_info.node_index, :]
            state['consumption_hist'] = list(consumer_states[:, IDX_CONSUMER_LATEST_CONSUMPTIONS])
            state['pending_order'] = list(cur_metrics["products"][product_unit_id]["pending_order_daily"])

        return

    def _update_consumer_features(
        self,
        state: dict,
        entity: SupplyChainEntity,
        cur_metrics: dict,
        storage_product_quantity: Dict[int, List[int]],
        facility_in_transit_orders: Dict[int, List[int]],
    ) -> None:
        if entity.skus is None:
            return

        # state['consumer_in_transit_orders'] = facility_in_transit_orders[entity.facility_id]

        # entity.skus.id -> SkuInfo.id -> unit.product_id
        state['inventory_in_stock'] = storage_product_quantity[entity.facility_id][
            self._global_sku_id2idx[entity.skus.id]
        ]
        state['inventory_in_transit'] = facility_in_transit_orders[entity.facility_id][
            self._global_sku_id2idx[entity.skus.id]
        ]

        pending_order = cur_metrics["facilities"][entity.facility_id]["pending_order"]

        if pending_order is not None:
            state['inventory_in_distribution'] = pending_order[entity.skus.id]

        state['inventory_estimated'] = (
            state['inventory_in_stock'] + state['inventory_in_transit'] - state['inventory_in_distribution']
        )

        state['is_over_stock'] = int(state['inventory_estimated'] >= 0.5 * state['storage_capacity'])
        state['is_out_of_stock'] = int(state['inventory_estimated'] <= 0)

        service_index = state['service_level']

        if service_index not in self._service_index_ppf_cache:
            self._service_index_ppf_cache[service_index] = st.norm.ppf(service_index)

        ppf = self._service_index_ppf_cache[service_index]

        state['inventory_rop'] = (
            state['max_vlt'] * state['sale_mean'] + np.sqrt(state['max_vlt']) * state['sale_std'] * ppf
        )

        state['is_below_rop'] = int(state['inventory_estimated'] < state['inventory_rop'])

        return
