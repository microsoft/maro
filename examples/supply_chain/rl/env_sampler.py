# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import scipy.stats as st

from maro.event_buffer import CascadeEvent
from maro.rl.policy import AbsPolicy, RLPolicy
from maro.rl.rollout import AbsAgentWrapper, AbsEnvSampler, CacheElement, SimpleAgentWrapper
from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import (
    ConsumerAction, ConsumerUnit, ManufactureAction, ManufactureUnit, ProductUnit
)
from maro.simulator.scenarios.supply_chain.actions import SupplyChainAction
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SkuInfo, SkuMeta, SupplyChainEntity, VendorLeadingTimeInfo

from examples.supply_chain.common.balance_calculator import BalanceSheetCalculator

from .agent_state import serialize_state, SCAgentStates
from .config import (
    distribution_features, env_conf, seller_features, workflow_settings,
    IDX_DISTRIBUTION_PENDING_PRODUCT_QUANTITY, IDX_DISTRIBUTION_PENDING_ORDER_NUMBER,
    IDX_SELLER_TOTAL_DEMAND, IDX_SELLER_SOLD, IDX_SELLER_DEMAND,
)
from .policies import agent2policy, trainable_policies


class SCEnvSampler(AbsEnvSampler):
    def __init__(
        self,
        get_env: Callable[[], Env],
        policy_creator: Dict[str, Callable[[str], AbsPolicy]],
        agent2policy: Dict[Any, str],  # {agent_name: policy_name}
        trainable_policies: List[str] = None,
        agent_wrapper_cls: Type[AbsAgentWrapper] = SimpleAgentWrapper,
        reward_eval_delay: int = 0,
        get_test_env: Callable[[], Env] = None,
    ) -> None:
        super().__init__(
            get_env,
            policy_creator,
            agent2policy,
            trainable_policies=trainable_policies,
            agent_wrapper_cls=agent_wrapper_cls,
            reward_eval_delay=reward_eval_delay,
            get_test_env=get_test_env,
        )
        self._env_settings: dict = workflow_settings

        self._balance_calculator: BalanceSheetCalculator = BalanceSheetCalculator(self._learn_env)

        self._configs: dict = self._learn_env.configs

        self._entity_dict: Dict[int, SupplyChainEntity] = {
            entity.id: entity
            for entity in self._learn_env.business_engine.get_entity_list()
        }

        self._summary: dict = self._learn_env.summary['node_mapping']

        # Key: Unit id; Value: (unit.data_model_name, unit.data_model_index, unit.facility.id, SkuInfo)
        self._units_mapping: Dict[int, Tuple[str, int, int, SkuInfo]] = self._summary["unit_mapping"]

        self._sku_metas: Dict[int, SkuMeta] = self._summary["skus"]
        self._global_sku_id2idx: Dict[int, int] = {
            sku_id: idx
            for idx, sku_id in enumerate(self._sku_metas.keys())
        }
        self._sku_number: int = len(self._sku_metas)

        self._facility_info_dict: Dict[int, FacilityInfo] = self._summary["facilities"]

        self._unit2product_unit: Dict[int, int] = self._get_unit2product_unit(self._facility_info_dict)

        # Key 1: Facility id; Key 2: Product id; Value: Index in product list
        self._product_id2idx: Dict[int, Dict[int, int]] = self._get_product_id2idx(self._facility_info_dict)

        # Key: Consumer unit id; Value: corresponding product id.
        self._consumer2product_id: Dict[int, int] = self._get_consumer2product_id(self._facility_info_dict)
        # Key: Consumer unit it; Value: source facility id list.
        self._consumer2source_facilities: Dict[int, List[int]] = self._get_consumer2source_facilities(
            self._facility_info_dict
        )

        self._cur_metrics: dict = self._learn_env.metrics

        # Key: facility/unit id; Value: (balance, reward)
        self._cur_balance_sheet_reward: Dict[int, Tuple[float, float]] = None

        # States of current tick, extracted from snapshot list.
        self._cur_distribution_states = None
        self._cur_consumer_states = None
        self._cur_seller_states = None

        # Key: facility id; List Index: sku idx; Value: in transition product quantity.
        self._facility_in_transit_orders: Dict[int, List[int]] = {}

        self._facility_product_utilization: Dict[int, int] = {}

        # Key: facility id
        self._storage_product_quantity: Dict[int, List[int]] = defaultdict(lambda: [0] * self._sku_number)

        # Key: service level; Value: the cached return value of Percentage Point Function.
        self._service_index_ppf_cache: Dict[float, float] = {}

        self._agent_states: SCAgentStates = SCAgentStates(
            entity_dict=self._entity_dict,
            facility_info_dict=self._facility_info_dict,
            global_sku_id2idx=self._global_sku_id2idx,
            sku_number=self._sku_number,
            max_src_per_facility=self._summary["max_sources_per_facility"],
            max_price=self._summary["max_price"],
            settings=self._env_settings,
        )

    def _get_unit2product_unit(self, facility_info_dict: Dict[int, FacilityInfo]) -> Dict[int, int]:
        unit2product: Dict[int, int] = {}
        for facility_info in facility_info_dict.values():
            for product_info in facility_info.products_info.values():
                for unit in (
                    product_info, product_info.seller_info, product_info.consumer_info, product_info.manufacture_info
                ):
                    if unit is not None:
                        unit2product[unit.id] = product_info.id
        return unit2product

    def _get_product_id2idx(self, facility_info_dict: Dict[int, FacilityInfo]) -> Dict[int, Dict[int, int]]:
        # Key 1: facility id; Key 2: product id; Value: index in product list.
        product_id2idx: Dict[int, Dict[int, int]] = defaultdict(dict)

        for facility_id, facility_info in facility_info_dict.items():
            if facility_info.storage_info is not None:
                for i, pid in enumerate(facility_info.storage_info.product_list):
                    product_id2idx[facility_id][pid] = i

        return product_id2idx

    def _get_consumer2product_id(self, facility_info_dict: Dict[int, FacilityInfo]) -> Dict[int, int]:
        consumer2product_id: Dict[int, int] = {}

        for facility_info in facility_info_dict.values():
            for product_id, product in facility_info.products_info.items():
                if product.consumer_info:
                    consumer2product_id[product.consumer_info.id] = product_id

        return consumer2product_id

    def _get_consumer2source_facilities(self, facility_info_dict: Dict[int, FacilityInfo]) -> Dict[int, List[int]]:
        consumer2source_facilities: Dict[int, List[int]] = {}

        for facility_info in facility_info_dict.values():
            for product in facility_info.products_info.values():
                if product.consumer_info:
                    consumer2source_facilities[product.consumer_info.id] = product.consumer_info.source_facility_id_list

        return consumer2source_facilities

    def _get_reward_for_entity(self, entity: SupplyChainEntity, bwt: Tuple[float, float]) -> float:
        if entity.class_type == ConsumerUnit:
            return np.float32(bwt[1]) / np.float32(self._env_settings["reward_normalization"])
        else:
            return .0

    def get_or_policy_state(self, entity: SupplyChainEntity) -> np.ndarray:
        # TODO: check the correctness of the implementation
        if entity.skus is None:
            return np.array([1])

        np_state, offsets = [0], [1]

        def extend_state(value: list) -> None:
            np_state.extend(value)
            offsets.append(len(np_state))

        product_unit_id = entity.id if entity.class_type == ProductUnit else entity.parent_id

        product_index = self._balance_calculator.product_id2idx.get(product_unit_id, None)
        unit_storage_cost = self._balance_calculator.products[product_index][4] if product_index is not None else 0

        product_metrics = self._cur_metrics["products"].get(product_unit_id, None)
        extend_state([product_metrics["sale_mean"] if product_metrics else 0])
        extend_state([product_metrics["sale_std"] if product_metrics else 0])

        extend_state([unit_storage_cost])
        extend_state([1])
        facility_info = self._facility_info_dict[entity.facility_id]
        product_info = facility_info.products_info[entity.skus.id]
        if product_info.consumer_info is not None:
            idx = product_info.consumer_info.node_index
            np_state[-1] = self._learn_env.snapshot_list["consumer"][
                self._learn_env.tick:idx:"order_cost"
            ].flatten()[0]

        be = self._env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)
        extend_state([facility_info.storage_info.config[0].capacity])
        extend_state(self._storage_product_quantity[entity.facility_id])
        extend_state(self._facility_in_transit_orders[entity.facility_id])
        extend_state([self._product_id2idx[entity.facility_id][entity.skus.id] + 1])  # TODO: check +1 or not
        extend_state([be.world.get_facility_by_id(entity.facility_id).get_max_vlt(entity.skus.id)])
        extend_state([entity.skus.service_level])
        return np.array(np_state + offsets)

    def get_rl_policy_state(self, entity_id: int, entity: SupplyChainEntity) -> np.ndarray:
        state = self._agent_states.templates[entity_id]

        self._update_global_features(state)
        self._update_facility_features(state, entity)
        self._update_storage_features(state, entity)
        self._update_sale_features(state, entity)
        self._update_distribution_features(state, entity)
        self._update_consumer_features(state, entity)

        np_state = serialize_state(state)
        return np_state

    def _get_entity_state(self, entity_id: int) -> np.ndarray:
        entity = self._entity_dict[entity_id]

        if isinstance(self._policy_dict[self._agent2policy[entity_id]], RLPolicy):
            return self.get_rl_policy_state(entity_id, entity)
        else:
            return self.get_or_policy_state(entity)

    def _get_global_and_agent_state(self, event: CascadeEvent, tick: int = None) -> tuple:
        """Update the status variables first, then call the state shaper for each agent."""
        if tick is None:
            tick = self._learn_env.tick
        else:
            # To make sure the usage of metrics is correct, the tick should be same to the current env tick.
            assert tick == self._learn_env.tick

        self._cur_metrics = self._learn_env.metrics

        # Get balance info of current tick from balance calculator.
        self._cur_balance_sheet_reward = self._balance_calculator.calc_and_update_balance_sheet(tick=tick)

        # Get distribution features of current tick from snapshot list.
        self._cur_distribution_states = self._learn_env.snapshot_list["distribution"][
            tick::distribution_features
        ].flatten().reshape(-1, len(distribution_features)).astype(np.int)

        # Get consumer features of specific ticks from snapshot list.
        consumption_hist_ticks = [tick - i for i in range(self._env_settings['consumption_hist_len'] - 1, -1, -1)]
        self._cur_consumer_states = self._learn_env.snapshot_list["consumer"][
            consumption_hist_ticks::"latest_consumptions"
        ].flatten().reshape(-1, len(self._learn_env.snapshot_list["consumer"]))

        # Get seller features of specific ticks from snapshot list.
        sale_hist_ticks = [tick - i for i in range(self._env_settings['sale_hist_len'] - 1, -1, -1)]
        self._cur_seller_states = self._learn_env.snapshot_list["seller"][
            sale_hist_ticks::seller_features
        ].astype(np.int)

        # 1. Update storage product quantity info.
        # 2. Update facility product utilization info.
        # 3. Update facility in transition order quantity info.
        for facility_id, facility_info in self._facility_info_dict.items():
            # Reset for each step
            self._facility_product_utilization[facility_id] = 0
            self._facility_in_transit_orders[facility_id] = [0] * self._sku_number

            product_quantities = self._learn_env.snapshot_list["storage"][
                tick:facility_info.storage_info.node_index:"product_quantity"
            ].flatten().astype(np.int)

            for pid, index in self._product_id2idx[facility_id].items():
                product_quantity = product_quantities[index]

                self._storage_product_quantity[facility_id][self._global_sku_id2idx[pid]] = product_quantity
                self._facility_product_utilization[facility_id] += product_quantity

            for sku_id, quantity in self._cur_metrics['facilities'][facility_id]["in_transit_orders"].items():
                self._facility_in_transit_orders[facility_id][self._global_sku_id2idx[sku_id]] = quantity

        state = {
            id_: self._get_entity_state(id_)
            for id_ in self._agent2policy.keys()
        }
        return None, state

    def _get_reward(self, env_action_dict: Dict[Any, object], event: object, tick: int) -> Dict[Any, float]:
        # get related product, seller, consumer, manufacture unit id
        # NOTE: this mapping does not contain facility id, so if id is not exist, then means it is a facility
        self._cur_balance_sheet_reward = self._balance_calculator.calc_and_update_balance_sheet(tick=tick)

        return {
            unit_id: self._get_reward_for_entity(self._entity_dict[unit_id], bwt)
            for unit_id, bwt in self._cur_balance_sheet_reward.items()
            if unit_id in self._agent2policy
        }

    def _translate_to_env_action(self, action_dict: Dict[Any, np.ndarray], event: object) -> Dict[Any, object]:
        env_action_dict: Dict[int, SupplyChainAction] = {}

        for agent_id, action in action_dict.items():
            entity_id = agent_id
            env_action: Optional[SupplyChainAction] = None

            # Consumer action
            if issubclass(self._entity_dict[agent_id].class_type, ConsumerUnit):
                product_id: int = self._consumer2product_id.get(entity_id, 0)
                product_unit_id: int = self._unit2product_unit[entity_id]

                # TODO: vehicle type selection and source selection
                facility_info: FacilityInfo = self._facility_info_dict[self._entity_dict[entity_id].facility_id]
                vlt_info_cadidates: List[VendorLeadingTimeInfo] = [
                    vlt_info
                    for vlt_info in facility_info.upstream_vlt_infos[product_id]
                    if any([
                        self._env_settings["default_vehicle_type"] == None,
                        vlt_info.vehicle_type == self._env_settings["default_vehicle_type"],
                    ])
                ]

                if len(vlt_info_cadidates):
                    src_f_id = vlt_info_cadidates[0].src_facility.id
                    vehicle_type = vlt_info_cadidates[0].vehicle_type

                    try:
                        action_quantity = int(int(action) * self._cur_metrics["products"][product_unit_id]["sale_mean"])
                    except ValueError:
                        action_quantity = 0

                    # Ignore 0 quantity to reduce action number
                    if action_quantity:
                        env_action = ConsumerAction(entity_id, product_id, src_f_id, action_quantity, vehicle_type)

            # Manufacture action
            elif issubclass(self._entity_dict[agent_id].class_type, ManufactureUnit):
                sku = self._units_mapping[entity_id][3]
                if sku.production_rate:
                    env_action = ManufactureAction(id=entity_id, production_rate=int(sku.production_rate))

            if env_action:
                env_action_dict[agent_id] = env_action

        return env_action_dict

    def _update_facility_features(self, state: dict, entity: SupplyChainEntity) -> None:
        state['is_positive_balance'] = 1 if self._balance_calculator.accumulated_balance_sheet[entity.id] > 0 else 0

    def _update_storage_features(self, state: dict, entity: SupplyChainEntity) -> None:
        state['storage_levels'] = self._storage_product_quantity[entity.facility_id]
        state['storage_utilization'] = self._facility_product_utilization[entity.facility_id]

    def _update_sale_features(self, state: dict, entity: SupplyChainEntity) -> None:
        if entity.class_type not in {ConsumerUnit, ProductUnit}:
            return

        # Get product unit id for current agent.
        product_unit_id = entity.id if entity.class_type == ProductUnit else entity.parent_id
        product_metrics = self._cur_metrics["products"][product_unit_id]

        state['sale_mean'] = product_metrics["sale_mean"]
        state['sale_std'] = product_metrics["sale_std"]

        product_info = self._facility_info_dict[entity.facility_id].products_info[entity.skus.id]

        if product_info.seller_info is not None:
            seller_states = self._cur_seller_states[:, product_info.seller_info.node_index, :]

            # For total demand, we need latest one.
            state['total_backlog_demand'] = seller_states[:, IDX_SELLER_TOTAL_DEMAND][-1][0]
            state['sale_hist'] = list(seller_states[:, IDX_SELLER_SOLD].flatten())
            state['backlog_demand_hist'] = list(seller_states[:, IDX_SELLER_DEMAND])

        else:
            state['sale_gamma'] = state['sale_mean']

        if product_info.consumer_info is not None:
            state['consumption_hist'] = list(self._cur_consumer_states[:, product_info.consumer_info.node_index])
            state['pending_order'] = list(product_metrics["pending_order_daily"])

    def _update_distribution_features(self, state: dict, entity: SupplyChainEntity) -> None:
        distribution_info = self._facility_info_dict[entity.facility_id].distribution_info
        if distribution_info is not None:
            dist_states = self._cur_distribution_states[distribution_info.node_index]
            state['distributor_in_transit_orders_qty'] = dist_states[IDX_DISTRIBUTION_PENDING_PRODUCT_QUANTITY]
            state['distributor_in_transit_orders'] = dist_states[IDX_DISTRIBUTION_PENDING_ORDER_NUMBER]

    def _update_consumer_features(self, state: dict, entity: SupplyChainEntity) -> None:
        if entity.skus is None:
            return

        state['consumer_in_transit_orders'] = self._facility_in_transit_orders[entity.facility_id]

        # entity.skus.id -> SkuInfo.id -> unit.product_id
        state['inventory_in_stock'] = self._storage_product_quantity[entity.facility_id][self._global_sku_id2idx[entity.skus.id]]
        state['inventory_in_transit'] = state['consumer_in_transit_orders'][self._global_sku_id2idx[entity.skus.id]]

        pending_order = self._cur_metrics["facilities"][entity.facility_id]["pending_order"]

        if pending_order is not None:
            state['inventory_in_distribution'] = pending_order[entity.skus.id]

        state['inventory_estimated'] = (
            state['inventory_in_stock'] + state['inventory_in_transit'] - state['inventory_in_distribution']
        )

        if state['inventory_estimated'] >= 0.5 * state['storage_capacity']:
            state['is_over_stock'] = 1

        if state['inventory_estimated'] <= 0:
            state['is_out_of_stock'] = 1

        service_index = state['service_level']

        if service_index not in self._service_index_ppf_cache:
            self._service_index_ppf_cache[service_index] = st.norm.ppf(service_index)

        ppf = self._service_index_ppf_cache[service_index]

        state['inventory_rop'] = (
            state['max_vlt'] * state['sale_mean'] + np.sqrt(state['max_vlt']) * state['sale_std'] * ppf
        )

        if state['inventory_estimated'] < state['inventory_rop']:
            state['is_below_rop'] = 1

    def _update_global_features(self, state) -> None:
        state["global_time"] = self._learn_env.tick

    def _post_step(self, cache_element: CacheElement, reward: Dict[Any, float]) -> None:
        tick = cache_element.tick
        total_sold = self._learn_env.snapshot_list["seller"][tick::"total_sold"].reshape(-1)
        total_demand = self._learn_env.snapshot_list["seller"][tick::"total_demand"].reshape(-1)
        self._info["sold"] = total_sold
        self._info["demand"] = total_demand
        self._info["sold/demand"] = self._info["sold"] / self._info["demand"]

    def _post_eval_step(self, cache_element: CacheElement, reward: Dict[Any, float]) -> None:
        self._post_step(cache_element, reward)


def env_sampler_creator(policy_creator) -> SCEnvSampler:
    return SCEnvSampler(
        get_env=lambda: Env(**env_conf),
        policy_creator=policy_creator,
        agent2policy=agent2policy,
        trainable_policies=trainable_policies,
    )
