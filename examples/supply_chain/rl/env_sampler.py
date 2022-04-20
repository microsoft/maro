# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from maro.event_buffer import CascadeEvent
from maro.rl.policy import AbsPolicy, RLPolicy
from maro.rl.rollout import AbsAgentWrapper, AbsEnvSampler, CacheElement, SimpleAgentWrapper
from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import (
    ConsumerAction, ConsumerUnit, ManufactureAction, ManufactureUnit
)
from maro.simulator.scenarios.supply_chain.actions import SupplyChainAction
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SkuInfo, SkuMeta, SupplyChainEntity, VendorLeadingTimeInfo

from examples.supply_chain.common.balance_calculator import BalanceSheetCalculator

from .algorithms.rule_based import ConsumerBasePolicy
from .config import (
    consumer_features, distribution_features, env_conf, seller_features, workflow_settings,
)
from .or_agent_state import ScOrAgentStates
from .policies import agent2policy, trainable_policies
from .rl_agent_state import ScRlAgentStates


def get_unit2product_unit(facility_info_dict: Dict[int, FacilityInfo]) -> Dict[int, int]:
    unit2product: Dict[int, int] = {}
    for facility_info in facility_info_dict.values():
        for product_info in facility_info.products_info.values():
            for unit in (
                product_info, product_info.seller_info, product_info.consumer_info, product_info.manufacture_info
            ):
                if unit is not None:
                    unit2product[unit.id] = product_info.id
    return unit2product


def get_product_id2idx(facility_info_dict: Dict[int, FacilityInfo]) -> Dict[int, Dict[int, int]]:
    # Key 1: facility id; Key 2: product id; Value: index in product list.
    product_id2idx: Dict[int, Dict[int, int]] = defaultdict(dict)

    for facility_id, facility_info in facility_info_dict.items():
        if facility_info.storage_info is not None:
            for i, pid in enumerate(facility_info.storage_info.product_list):
                product_id2idx[facility_id][pid] = i

    return product_id2idx


def get_consumer2product_id(facility_info_dict: Dict[int, FacilityInfo]) -> Dict[int, int]:
    consumer2product_id: Dict[int, int] = {}

    for facility_info in facility_info_dict.values():
        for product_id, product in facility_info.products_info.items():
            if product.consumer_info:
                consumer2product_id[product.consumer_info.id] = product_id

    return consumer2product_id


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

        self._unit2product_unit: Dict[int, int] = get_unit2product_unit(self._facility_info_dict)

        # Key 1: Facility id; Key 2: Product id; Value: Index in product list
        self._product_id2idx: Dict[int, Dict[int, int]] = get_product_id2idx(self._facility_info_dict)

        # Key: Consumer unit id; Value: corresponding product id.
        self._consumer2product_id: Dict[int, int] = get_consumer2product_id(self._facility_info_dict)

        self._cur_metrics: dict = self._learn_env.metrics

        # Key: facility/unit id; Value: (balance, reward)
        self._cur_balance_sheet_reward: Dict[int, Tuple[float, float]] = {}

        # States of current tick, extracted from snapshot list.
        self._cur_distribution_states: Optional[np.ndarray] = None
        self._cur_seller_hist_states: Optional[np.ndarray] = None
        self._cur_consumer_hist_states: Optional[np.ndarray] = None

        # Key: facility id; List Index: sku idx; Value: in transition product quantity.
        self._facility_in_transit_orders: Dict[int, List[int]] = {}

        self._facility_product_utilization: Dict[int, int] = {}

        # Key: facility id
        self._storage_product_quantity: Dict[int, List[int]] = defaultdict(lambda: [0] * self._sku_number)

        self._rl_agent_states: ScRlAgentStates = ScRlAgentStates(
            entity_dict=self._entity_dict,
            facility_info_dict=self._facility_info_dict,
            global_sku_id2idx=self._global_sku_id2idx,
            sku_number=self._sku_number,
            max_src_per_facility=self._summary["max_sources_per_facility"],
            max_price=self._summary["max_price"],
            settings=self._env_settings,
        )

        self._storage_unit_cost_dict: Optional[Dict[int, Dict[int, int]]] = None
        self._storage_capacity_dict: Optional[Dict[int, Dict[int, int]]] = None

        self._or_agent_states: ScOrAgentStates = ScOrAgentStates(
            entity_dict=self._entity_dict,
            facility_info_dict=self._facility_info_dict,
            global_sku_id2idx=self._global_sku_id2idx,
        )

    def _get_storage_unit_cost_and_capacity_info(self) -> Tuple[Dict[int, Dict[int, int]], Dict[int, Dict[int, int]]]:
        # Key1: storage node index; Key2: product id/sku id; Value: storage unit cost.
        storage_unit_cost_dict: Dict[int, Dict[int, int]] = defaultdict(dict)
        # Key1: storage node index; Key2: product id/sku id; Value: sub storage capacity.
        storage_capacity_dict: Dict[int, Dict[int, int]] = defaultdict(dict)

        storage_snapshots = self._learn_env.snapshot_list["storage"]
        for node_index in range(len(storage_snapshots)):
            storage_unit_cost_list = storage_snapshots[0:node_index:"unit_storage_cost"].flatten()
            storage_capacity_list = storage_snapshots[0:node_index:"capacity"].flatten().astype(int)
            product_storage_index_list = storage_snapshots[0:node_index:"product_storage_index"].flatten().astype(int)
            product_id_list = storage_snapshots[0:node_index:"product_list"].flatten().astype(int)

            for product_id, sub_storage_idx in zip(product_id_list, product_storage_index_list):
                storage_unit_cost_dict[node_index][product_id] = storage_unit_cost_list[sub_storage_idx]
                storage_capacity_dict[node_index][product_id] = storage_capacity_list[sub_storage_idx]

        return storage_unit_cost_dict, storage_capacity_dict

    def _get_reward_for_entity(self, entity: SupplyChainEntity, bwt: Tuple[float, float]) -> float:
        if entity.class_type == ConsumerUnit:
            return np.float32(bwt[1]) / np.float32(self._env_settings["reward_normalization"])
        else:
            return .0

    def get_or_policy_state(self, entity: SupplyChainEntity) -> dict:
        if self._storage_unit_cost_dict is None:
            self._storage_unit_cost_dict, self._storage_capacity_dict = self._get_storage_unit_cost_and_capacity_info()

        state = self._or_agent_states._update_entity_state(
            entity_id=entity.id,
            storage_unit_cost_dict=self._storage_unit_cost_dict,
            storage_capacity_dict=self._storage_capacity_dict,
            product_metrics=self._cur_metrics["products"].get(self._unit2product_unit[entity.id], None),
            cur_consumer_hist_states=self._cur_consumer_hist_states,
            product_levels=self._storage_product_quantity[entity.facility_id],
            in_transit_order_quantity=self._facility_in_transit_orders[entity.facility_id],
        )

        return state

    def get_rl_policy_state(self, entity_id: int) -> np.ndarray:
        np_state = self._rl_agent_states.update_entity_state(
            entity_id=entity_id,
            tick=self._learn_env.tick,
            cur_metrics=self._cur_metrics,
            cur_distribution_states=self._cur_distribution_states,
            cur_seller_hist_states=self._cur_seller_hist_states,
            cur_consumer_hist_states=self._cur_consumer_hist_states,
            accumulated_balance=self._balance_calculator.accumulated_balance_sheet[entity_id],
            storage_product_quantity=self._storage_product_quantity,
            facility_product_utilization=self._facility_product_utilization,
            facility_in_transit_orders=self._facility_in_transit_orders,
        )
        return np_state

    def _get_entity_state(self, entity_id: int) -> Union[np.ndarray, dict, None]:
        entity = self._entity_dict[entity_id]

        if isinstance(self._policy_dict[self._agent2policy[entity_id]], RLPolicy):
            return self.get_rl_policy_state(entity_id)
        elif isinstance(self._policy_dict[self._agent2policy[entity_id]], ConsumerBasePolicy):
            return self.get_or_policy_state(entity)
        else:
            return None

    def _get_global_and_agent_state_impl(
        self, event: CascadeEvent, tick: int = None,
    ) -> Tuple[Union[None, np.ndarray, List[object]], Dict[Any, Union[np.ndarray, List[object]]]]:
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
        self._cur_consumer_hist_states = self._learn_env.snapshot_list["consumer"][
            consumption_hist_ticks::consumer_features
        ].reshape(self._env_settings['consumption_hist_len'], -1, len(consumer_features))

        # Get seller features of specific ticks from snapshot list.
        sale_hist_ticks = [tick - i for i in range(self._env_settings['sale_hist_len'] - 1, -1, -1)]
        self._cur_seller_hist_states = self._learn_env.snapshot_list["seller"][
            sale_hist_ticks::seller_features
        ].reshape(self._env_settings['sale_hist_len'], -1, len(seller_features)).astype(np.int)

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

    def _translate_to_env_action(
        self, action_dict: Dict[Any, Union[np.ndarray, List[object]]], event: object,
    ) -> Dict[Any, object]:
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
                vlt_info_candidates: List[VendorLeadingTimeInfo] = [
                    vlt_info
                    for vlt_info in facility_info.upstream_vlt_infos[product_id]
                    if any([
                        self._env_settings["default_vehicle_type"] is None,
                        vlt_info.vehicle_type == self._env_settings["default_vehicle_type"],
                    ])
                ]

                if len(vlt_info_candidates):
                    src_f_id = vlt_info_candidates[0].src_facility.id
                    vehicle_type = vlt_info_candidates[0].vehicle_type

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
