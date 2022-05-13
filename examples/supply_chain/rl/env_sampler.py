# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import csv
import os
import random
from collections import defaultdict
from os.path import dirname, join, realpath
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from examples.supply_chain.common.balance_calculator import BalanceSheetCalculator
from maro.event_buffer import CascadeEvent
from maro.rl.policy import RLPolicy
from maro.rl.rollout import AbsAgentWrapper, AbsEnvSampler, CacheElement, SimpleAgentWrapper
from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ConsumerAction, ConsumerUnit, ManufactureUnit
from maro.simulator.scenarios.supply_chain.actions import SupplyChainAction
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SkuInfo, SkuMeta, SupplyChainEntity, VendorLeadingTimeInfo
from maro.simulator.scenarios.supply_chain.parser import SupplyChainConfiguration

from .algorithms.rule_based import ConsumerBasePolicy
from .config import VehicleSelection, consumer_features, distribution_features, seller_features, workflow_settings
from .or_agent_state import ScOrAgentStates
from .rl_agent_state import ScRlAgentStates

OUTPUT_CSV_FOLDER = join(dirname(dirname(realpath(__file__))), "results")
os.makedirs(OUTPUT_CSV_FOLDER, exist_ok=True)
OUTPUT_CSV_PATH = join(OUTPUT_CSV_FOLDER, "baseline.csv")


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
        learn_env: Env,
        test_env: Env,
        agent_wrapper_cls: Type[AbsAgentWrapper] = SimpleAgentWrapper,
        reward_eval_delay: int = 0,
    ) -> None:
        super().__init__(learn_env, test_env, agent_wrapper_cls, reward_eval_delay)

        self._env_settings: dict = workflow_settings

        business_engine = self._learn_env.business_engine
        assert isinstance(business_engine, SupplyChainBusinessEngine)
        self._entity_dict: Dict[int, SupplyChainEntity] = {
            entity.id: entity
            for entity in business_engine.get_entity_list()
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

        self._configs = self._learn_env.configs  # TODO: Optimize type hint here
        assert isinstance(self._configs, SupplyChainConfiguration)
        self._policy_parameter: Dict[str, Any] = self._parse_policy_parameter(self._configs.policy_parameters)

        self._balance_calculator: BalanceSheetCalculator = BalanceSheetCalculator(self._learn_env)

        ########################################################################
        # Internal Variables. Would be updated and used.
        ########################################################################

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

        self._storage_capacity_dict: Optional[Dict[int, Dict[int, int]]] = None

        ########################################################################
        # State managers.
        ########################################################################

        self._rl_agent_states: ScRlAgentStates = ScRlAgentStates(
            entity_dict=self._entity_dict,
            facility_info_dict=self._facility_info_dict,
            global_sku_id2idx=self._global_sku_id2idx,
            sku_number=self._sku_number,
            max_src_per_facility=self._summary["max_sources_per_facility"],
            max_price_dict=self._policy_parameter["max_price"],
            settings=self._env_settings,
        )

        self._or_agent_states: ScOrAgentStates = ScOrAgentStates(
            entity_dict=self._entity_dict,
            facility_info_dict=self._facility_info_dict,
            global_sku_id2idx=self._global_sku_id2idx,
        )

    def _parse_policy_parameter(self, raw_info: dict) -> Dict[str, Any]:
        facility_name2id: Dict[str, int] = {
            facility_info.name: facility_id
            for facility_id, facility_info in self._facility_info_dict.items()
        }

        max_prices: Dict[int, float] = {}
        global_max_price: float = 0
        for facility_name, infos in raw_info.get("facilities", {}).items():
            if infos.get("max_price", None) is not None:
                max_price = float(infos["max_price"])
                max_prices[facility_name2id[facility_name]] = max_price
                global_max_price = max(global_max_price, max_price)

        # Set the global max price for the facilities whose max_price is not set.
        for facility_id in self._facility_info_dict.keys():
            if facility_id not in max_prices:
                max_prices[facility_id] = global_max_price

        policy_parameter: Dict[str, Any] = {
            "max_price": max_prices,
        }

        return policy_parameter

    def _get_storage_capacity_dict_info(self) -> Dict[int, Dict[int, int]]:
        # Key1: storage node index; Key2: product id/sku id; Value: sub storage capacity.
        storage_capacity_dict: Dict[int, Dict[int, int]] = defaultdict(dict)

        storage_snapshots = self._env.snapshot_list["storage"]
        for node_index in range(len(storage_snapshots)):
            storage_capacity_list = storage_snapshots[0:node_index:"capacity"].flatten().astype(int)
            product_storage_index_list = storage_snapshots[0:node_index:"product_storage_index"].flatten().astype(int)
            product_id_list = storage_snapshots[0:node_index:"product_list"].flatten().astype(int)

            for product_id, sub_storage_idx in zip(product_id_list, product_storage_index_list):
                storage_capacity_dict[node_index][product_id] = storage_capacity_list[sub_storage_idx]

        return storage_capacity_dict

    def _get_reward_for_entity(self, entity: SupplyChainEntity, bwt: Tuple[float, float]) -> float:
        if entity.class_type == ConsumerUnit:
            return np.float32(bwt[1]) / np.float32(self._env_settings["reward_normalization"])
        else:
            return .0

    def get_or_policy_state(self, entity: SupplyChainEntity) -> dict:
        if self._storage_capacity_dict is None:
            self._storage_capacity_dict = self._get_storage_capacity_dict_info()

        state = self._or_agent_states.update_entity_state(
            entity_id=entity.id,
            storage_capacity_dict=self._storage_capacity_dict,
            product_metrics=self._cur_metrics["products"].get(self._unit2product_unit[entity.id], None),
            product_levels=self._storage_product_quantity[entity.facility_id],
            in_transit_order_quantity=self._facility_in_transit_orders[entity.facility_id],
        )

        return state

    def get_rl_policy_state(self, entity_id: int) -> np.ndarray:
        np_state = self._rl_agent_states.update_entity_state(
            entity_id=entity_id,
            tick=self._env.tick,
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
            tick = self._env.tick
        else:
            # To make sure the usage of metrics is correct, the tick should be same to the current env tick.
            assert tick == self._env.tick

        self._balance_calculator.update_env(self._env)

        self._cur_metrics = self._env.metrics

        # Get balance info of current tick from balance calculator.
        self._cur_balance_sheet_reward = self._balance_calculator.calc_and_update_balance_sheet(tick=tick)

        # Get distribution features of current tick from snapshot list.
        self._cur_distribution_states = self._env.snapshot_list["distribution"][
            tick::distribution_features
        ].flatten().reshape(-1, len(distribution_features)).astype(np.int)

        # Get consumer features of specific ticks from snapshot list.
        consumption_hist_ticks = [tick - i for i in range(self._env_settings['consumption_hist_len'] - 1, -1, -1)]
        self._cur_consumer_hist_states = self._env.snapshot_list["consumer"][
            consumption_hist_ticks::consumer_features
        ].reshape(self._env_settings['consumption_hist_len'], -1, len(consumer_features))

        # Get seller features of specific ticks from snapshot list.
        sale_hist_ticks = [tick - i for i in range(self._env_settings['sale_hist_len'] - 1, -1, -1)]
        self._cur_seller_hist_states = self._env.snapshot_list["seller"][
            sale_hist_ticks::seller_features
        ].reshape(self._env_settings['sale_hist_len'], -1, len(seller_features)).astype(np.int)

        # 1. Update storage product quantity info.
        # 2. Update facility product utilization info.
        # 3. Update facility in transition order quantity info.
        for facility_id, facility_info in self._facility_info_dict.items():
            # Reset for each step
            self._facility_product_utilization[facility_id] = 0
            self._facility_in_transit_orders[facility_id] = [0] * self._sku_number

            if facility_info.storage_info.node_index is not None:
                product_quantities = self._env.snapshot_list["storage"][
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
        self._balance_calculator.update_env(self._env)
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
                vlt_info_candidates: List[VendorLeadingTimeInfo] = []
                facility_info: FacilityInfo = self._facility_info_dict[self._entity_dict[entity_id].facility_id]
                info_by_fid = facility_info.upstream_vlt_infos[product_id]
                if self._env_settings["default_vehicle_type"] is None:
                    vlt_info_candidates = [
                        info
                        for info_by_type in info_by_fid.values()
                        for info in info_by_type.values()
                    ]
                else:
                    vlt_info_candidates = [
                        info_by_type[self._env_settings["default_vehicle_type"]]
                        for info_by_type in info_by_fid.values()
                    ]

                if len(vlt_info_candidates):
                    vehicle_selection = self._env_settings["vehicle_selection_method"]
                    if vehicle_selection == VehicleSelection.FIRST_ONE:
                        vlt_info = vlt_info_candidates[0]
                    elif vehicle_selection == VehicleSelection.RANDOM:
                        vlt_info = random.choice(vlt_info_candidates)
                    elif vehicle_selection == VehicleSelection.SHORTEST_LEADING_TIME:
                        vlt_info = min(vlt_info_candidates, key=lambda x: x.vlt)
                    elif vehicle_selection == VehicleSelection.CHEAPEST_TOTAL_COST:
                        # As the product cost and order base cost are only related to product quantity,
                        # the transportation cost is the difference of different vehicle type selections.
                        vlt_info = min(vlt_info_candidates, key=lambda x: x.unit_transportation_cost * (x.vlt + 1))
                    else:
                        raise Exception(f"Vehicle Selection method undefined: {vehicle_selection}")

                    src_f_id = vlt_info.src_facility.id
                    vehicle_type = vlt_info.vehicle_type

                    try:
                        action_quantity = int(int(action) * self._cur_metrics["products"][product_unit_id]["sale_mean"])
                    except ValueError:
                        action_quantity = 0

                    # Ignore 0 quantity to reduce action number
                    if action_quantity:
                        env_action = ConsumerAction(entity_id, product_id, src_f_id, action_quantity, vehicle_type)

            # Manufacture action
            elif issubclass(self._entity_dict[agent_id].class_type, ManufactureUnit):
                pass

            if env_action:
                env_action_dict[agent_id] = env_action

        return env_action_dict

    def _post_step(self, cache_element: CacheElement, reward: Dict[Any, float]) -> None:
        tick = cache_element.tick
        total_sold = self._env.snapshot_list["seller"][tick::"total_sold"].reshape(-1)
        total_demand = self._env.snapshot_list["seller"][tick::"total_demand"].reshape(-1)
        self._info["sold"] = total_sold
        self._info["demand"] = total_demand
        self._info["sold/demand"] = self._info["sold"] / self._info["demand"]

    def _post_eval_step(self, cache_element: CacheElement, reward: Dict[Any, float]) -> None:
        self._post_step(cache_element, reward)

    def post_collect(self, info_list: list, ep: int) -> None:
        with open(OUTPUT_CSV_PATH, "a") as fp:
            writer = csv.writer(fp, delimiter=' ')
            for info in info_list:
                writer.writerow([ep, info["sold"], info["demand"], info["sold/demand"]])
            # print the average env metric
            if len(info_list) > 1:
                metric_keys, num_envs = info_list[0].keys(), len(info_list)
                avg = {key: sum(info[key] for info in info_list) / num_envs for key in metric_keys}
                writer.writerow([ep, avg["sold"], avg["demand"], avg["sold/demand"]])

    def post_evaluate(self, info_list: list, ep: int) -> None:
        self.post_collect(info_list, ep)
