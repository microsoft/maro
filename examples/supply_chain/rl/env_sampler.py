# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Type

import numpy as np
import scipy.stats as st

from maro.event_buffer import CascadeEvent
from maro.rl.policy import AbsPolicy, RLPolicy
from maro.rl.rollout import AbsAgentWrapper, AbsEnvSampler, CacheElement, SimpleAgentWrapper
from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import (
    ConsumerAction, ConsumerUnit, ManufactureAction, ManufactureUnit, ProductUnit
)
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.world import SupplyChainEntity

from examples.supply_chain.common.balance_calculator import BalanceSheetCalculator

from .config import distribution_features, env_conf, seller_features
from .env_helper import STORAGE_INFO
from .policies import agent2policy, trainable_policies
from .state_template import keys_in_state, STATE_TEMPLATE, workflow_settings


def _serialize_state(state: dict) -> np.ndarray:
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
            get_env, policy_creator, agent2policy,
            trainable_policies=trainable_policies,
            agent_wrapper_cls=agent_wrapper_cls,
            reward_eval_delay=reward_eval_delay,
            get_test_env=get_test_env,
        )

        self._agent2policy = agent2policy
        self._entity_dict = {entity.id: entity for entity in self._learn_env.business_engine.get_entity_list()}
        self._balance_calculator = BalanceSheetCalculator(self._learn_env)
        self._cur_balance_sheet_reward = None

        self._summary = self._learn_env.summary['node_mapping']
        self._configs = self._learn_env.configs
        self._units_mapping = self._summary["unit_mapping"]

        self._sku_number = len(self._summary["skus"]) + 1  # ?: Why + 1?
        self._max_sources_per_facility = self._summary["max_sources_per_facility"]

        # state for each tick
        self._cur_metrics = self._learn_env.metrics
        # cache for ppf value.
        self._service_index_ppf_cache = {}

        # facility id -> in_transit_orders
        self._facility_in_transit_orders = {}
        # current distribution states
        self._cur_distribution_states = None
        # current consumer states
        self._cur_consumer_states = None
        # current seller states
        self._cur_seller_states = None

        self._stock_status = {}
        self._demand_status = {}
        # key: product unit id, value: number
        self._orders_from_downstreams = {}
        self._consumer_orders = {}
        self._order_in_transit_status = {}
        self._order_to_distribute_status = {}

        self._storage_info = STORAGE_INFO
        self._state_template = STATE_TEMPLATE

        self._env_settings = workflow_settings

    def _get_reward_for_entity(self, entity: SupplyChainEntity, bwt: list) -> float:
        if entity.class_type == ConsumerUnit:
            return np.float32(bwt[1]) / np.float32(self._env_settings["reward_normalization"])
        else:
            return .0

    def get_or_policy_state(self, state: dict, entity: SupplyChainEntity) -> np.ndarray:
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

        facility = self._storage_info["facility_levels"][entity.facility_id]
        extend_state([unit_storage_cost])
        extend_state([1])
        product_info = facility[entity.skus.id]
        if "consumer" in product_info:
            idx = product_info["consumer"].node_index
            np_state[-1] = self._learn_env.snapshot_list["consumer"][
                self._learn_env.tick:idx:"order_cost"
            ].flatten()[0]

        extend_state([facility['storage'].config[0].capacity])
        extend_state(self._storage_info["storage_product_num"][entity.facility_id])
        extend_state(self._facility_in_transit_orders[entity.facility_id])
        extend_state([self._storage_info["storage_product_indexes"][entity.facility_id][entity.skus.id] + 1])
        extend_state([entity.skus.vlt])
        extend_state([entity.skus.service_level])
        return np.array(np_state + offsets)

    def get_rl_policy_state(self, state: dict, entity: SupplyChainEntity) -> np.ndarray:
        self._update_facility_features(state, entity)
        self._update_storage_features(state, entity)
        # bom do not need to update
        # self._add_bom_features(state, entity)
        self._update_distribution_features(state, entity)
        self._update_sale_features(state, entity)
        # vlt do not need to update
        # self._update_vlt_features(state, entity)
        self._update_consumer_features(state, entity)
        # self._add_price_features(state, entity)
        self._update_global_features(state)

        self._stock_status[entity.id] = state['inventory_in_stock']

        self._demand_status[entity.id] = state['sale_hist'][-1]

        self._order_in_transit_status[entity.id] = state['inventory_in_transit']

        self._order_to_distribute_status[entity.id] = state['distributor_in_transit_orders_qty']

        np_state = _serialize_state(state)
        return np_state

    def _get_state_shaper(self, entity_id: int):
        if isinstance(self._policy_dict[self._agent2policy[entity_id]], RLPolicy):
            return self.get_rl_policy_state
        else:
            return self.get_or_policy_state

    def _get_global_and_agent_state(self, event: CascadeEvent, tick: int = None) -> tuple:
        if tick is None:
            tick = self._learn_env.tick
        settings: dict = self._env_settings
        consumption_hist_len = settings['consumption_hist_len']
        hist_len = settings['sale_hist_len']
        consumption_ticks = [tick - i for i in range(consumption_hist_len - 1, -1, -1)]
        hist_ticks = [tick - i for i in range(hist_len - 1, -1, -1)]

        self._cur_balance_sheet_reward = self._balance_calculator.calc_and_update_balance_sheet(tick=tick)
        self._cur_metrics = self._learn_env.metrics

        self._cur_distribution_states = self._learn_env.snapshot_list["distribution"][
            tick::distribution_features
        ].flatten().reshape(-1, len(distribution_features)).astype(np.int)

        self._cur_consumer_states = self._learn_env.snapshot_list["consumer"][
            consumption_ticks::"latest_consumptions"
        ].flatten().reshape(-1, len(self._learn_env.snapshot_list["consumer"]))

        self._cur_seller_states = self._learn_env.snapshot_list["seller"][hist_ticks::seller_features].astype(np.int)

        # facility level states
        for facility_id in self._storage_info["facility_product_utilization"]:
            # reset for each step
            self._storage_info["facility_product_utilization"][facility_id] = 0

            in_transit_orders = self._cur_metrics['facilities'][facility_id]["in_transit_orders"]

            self._facility_in_transit_orders[facility_id] = [0] * self._sku_number

            for sku_id, number in in_transit_orders.items():
                self._facility_in_transit_orders[facility_id][sku_id] = number

        # calculate storage info first, then use it later to speed up.
        for facility_id, storage_index in self._storage_info["facility2storage"].items():
            product_quantities = self._learn_env.snapshot_list["storage"][
                tick:storage_index:"product_quantity"
            ].flatten().astype(np.int)

            for pid, index in self._storage_info["storage_product_indexes"][facility_id].items():
                product_quantity = product_quantities[index]

                self._storage_info["storage_product_num"][facility_id][pid] = product_quantity
                self._storage_info["facility_product_utilization"][facility_id] += product_quantity

        state = {
            id_: self._get_state_shaper(id_)(self._state_template[id_], entity)
            for id_, entity in self._entity_dict.items() if id_ in self._agent2policy
        }
        return None, state

    def _get_reward(self, env_action_dict: Dict[Any, object], event: object, tick: int) -> Dict[Any, float]:
        # get related product, seller, consumer, manufacture unit id
        # NOTE: this mapping does not contain facility id, so if id is not exist, then means it is a facility
        self._cur_balance_sheet_reward = self._balance_calculator.calc_and_update_balance_sheet(tick=tick)
        return {
            f_id: self._get_reward_for_entity(self._entity_dict[f_id], bwt)
            for f_id, bwt in self._cur_balance_sheet_reward.items() if f_id in self._agent2policy
        }

    def _translate_to_env_action(self, action_dict: Dict[Any, np.ndarray], event: object) -> Dict[Any, object]:
        # cache the sources for each consumer if not yet cached
        if not hasattr(self, "consumer2source"):
            self.consumer2source, self.consumer2product = {}, {}
            facility_info_dict: Dict[int, FacilityInfo] = self._learn_env.summary["node_mapping"]["facilities"]
            for facility_info in facility_info_dict.values():
                for product_id, product in facility_info.products_info.items():
                    if product.consumer_info:
                        self.consumer2source[product.consumer_info.id] = product.consumer_info.source_facility_id_list
                        self.consumer2product[product.consumer_info.id] = product_id

        env_action_dict = {}
        for agent_id, action in action_dict.items():
            # ignore facility to reduce action number
            if agent_id not in self._units_mapping:
                continue

            unit_id = agent_id

            # consumer action
            if issubclass(self._entity_dict[agent_id].class_type, ConsumerUnit):
                product_id = self.consumer2product.get(unit_id, 0)
                sources = self.consumer2source.get(unit_id, [])
                if sources:
                    source_id = sources[0]
                    product_unit_id = self._storage_info["unit2product"][unit_id][0]
                    try:
                        action_number = int(int(action) * self._cur_metrics["products"][product_unit_id]["sale_mean"])
                    except ValueError:
                        action_number = 0

                    # ignore 0 quantity to reduce action number
                    if action_number:
                        sku = self._units_mapping[unit_id][3]
                        env_action_dict[agent_id] = ConsumerAction(
                            unit_id, product_id, source_id, action_number, sku.vlt,
                        )
                        self._consumer_orders[product_unit_id] = action_number
                        self._orders_from_downstreams[
                            self._storage_info["facility_levels"][source_id][product_id]["skuproduct"].id
                        ] = action_number
            # manufacturer action
            elif issubclass(self._entity_dict[agent_id].class_type, ManufactureUnit):
                sku = self._units_mapping[unit_id][3]
                action = sku.production_rate
                # ignore invalid actions
                if action:
                    env_action_dict[agent_id] = ManufactureAction(id=unit_id, production_rate=float(action))

        return env_action_dict

    def _update_facility_features(self, state: dict, entity: SupplyChainEntity) -> None:
        state['is_positive_balance'] = 1 if self._balance_calculator.accumulated_balance_sheet[entity.id] > 0 else 0

    def _update_storage_features(self, state: dict, entity: SupplyChainEntity) -> None:
        state['storage_utilization'] = 0

        state['storage_levels'] = self._storage_info["storage_product_num"][entity.facility_id]
        state['storage_utilization'] = self._storage_info["facility_product_utilization"][entity.facility_id]

    def _update_sale_features(self, state: dict, entity: SupplyChainEntity) -> None:
        if entity.class_type not in {ConsumerUnit, ProductUnit}:
            return

        # Get product unit id for current agent.
        product_unit_id = entity.id if entity.class_type == ProductUnit else entity.parent_id
        product_metrics = self._cur_metrics["products"][product_unit_id]

        state['sale_mean'] = product_metrics["sale_mean"]
        state['sale_std'] = product_metrics["sale_std"]

        facility = self._storage_info["facility_levels"][entity.facility_id]
        product_info = facility[entity.skus.id]

        if "seller" not in product_info:
            # TODO: why gamma sale as mean?
            state['sale_gamma'] = state['sale_mean']

        if "consumer" in product_info:
            consumer_index = product_info["consumer"].node_index

            state['consumption_hist'] = list(self._cur_consumer_states[:, consumer_index])
            state['pending_order'] = list(product_metrics["pending_order_daily"])

        if "seller" in product_info:
            seller_index = product_info["seller"].node_index

            seller_states = self._cur_seller_states[:, seller_index, :]

            # For total demand, we need latest one.
            state['total_backlog_demand'] = seller_states[:, 0][-1][0]
            state['sale_hist'] = list(seller_states[:, 1].flatten())
            state['backlog_demand_hist'] = list(seller_states[:, 2])

    def _update_distribution_features(self, state: dict, entity: SupplyChainEntity) -> None:
        facility = self._storage_info["facility_levels"][entity.facility_id]
        distribution = facility.get("distribution", None)

        if distribution is not None:
            dist_states = self._cur_distribution_states[distribution.node_index]
            state['distributor_in_transit_orders'] = dist_states[1]
            state['distributor_in_transit_orders_qty'] = dist_states[0]

    def _update_consumer_features(self, state: dict, entity: SupplyChainEntity) -> None:
        if entity.skus is None:
            return

        state['consumer_in_transit_orders'] = self._facility_in_transit_orders[entity.facility_id]

        # FIX: we need plus 1 to this, as it is 0 based index, but we already aligned with 1 more
        # slot to use sku id as index ( 1 based).
        product_index = self._storage_info["storage_product_indexes"][entity.facility_id][entity.skus.id] + 1
        state['inventory_in_stock'] = self._storage_info["storage_product_num"][entity.facility_id][product_index]
        state['inventory_in_transit'] = state['consumer_in_transit_orders'][entity.skus.id]

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
        trainable_policies=trainable_policies
    )
