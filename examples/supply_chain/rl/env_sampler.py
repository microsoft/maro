# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import datetime
import json
import os
import pickle
import random
import shutil
import sys
import typing
from collections import defaultdict
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from maro.event_buffer import CascadeEvent
from maro.rl.policy import RLPolicy, RuleBasedPolicy
from maro.rl.rollout import AbsAgentWrapper, AbsEnvSampler, CacheElement, SimpleAgentWrapper
from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import (
    ConsumerAction,
    ConsumerUnit,
    ManufactureAction,
    ManufactureUnit,
    StoreProductUnit,
)
from maro.simulator.scenarios.supply_chain.actions import SupplyChainAction
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities import FacilityBase, FacilityInfo, OuterRetailerFacility
from maro.simulator.scenarios.supply_chain.objects import SkuInfo, SkuMeta, SupplyChainEntity, VendorLeadingTimeInfo
from maro.simulator.scenarios.supply_chain.parser import SupplyChainConfiguration
from maro.simulator.scenarios.supply_chain.units import DistributionUnitInfo, ProductUnit, StorageUnitInfo
from maro.utils.logger import LogFormat, Logger

from .algorithms.base_stock_policy import BaseStockPolicy
from .algorithms.rule_based import ConsumerMinMaxPolicy as ConsumerBaselinePolicy
from .config import (
    ALGO,
    IDX_CONSUMER_PURCHASED,
    IDX_PRODUCT_PRICE,
    IDX_SELLER_DEMAND,
    OR_NUM_CONSUMER_ACTIONS,
    TEAM_REWARD,
    VehicleSelection,
    consumer_features,
    distribution_features,
    env_conf,
    product_features,
    seller_features,
    test_env_conf,
    workflow_settings,
)
from .or_agent_state import ScOrAgentStates
from .rl_agent_state import ScRlAgentStates, serialize_state
from examples.supply_chain.common.balance_calculator import BalanceSheetCalculator
from examples.supply_chain.common.render_tools.plot_render import SimulationTracker
from examples.supply_chain.common.utils import get_attributes, get_list_attributes

if typing.TYPE_CHECKING:
    from maro.rl.rl_component.rl_component_bundle import RLComponentBundle


def get_unit2product_unit(facility_info_dict: Dict[int, FacilityInfo]) -> Dict[int, int]:
    unit2product: Dict[int, int] = {}
    for facility_info in facility_info_dict.values():
        for product_info in facility_info.products_info.values():
            for unit in (
                product_info,
                product_info.seller_info,
                product_info.consumer_info,
                product_info.manufacture_info,
            ):
                if unit is not None:
                    unit2product[unit.id] = product_info.id
    return unit2product


def get_sku_id2idx_in_product_list(facility_info_dict: Dict[int, FacilityInfo]) -> Dict[int, Dict[int, int]]:
    # Key 1: facility id; Key 2: product id; Value: index in product list.
    sku_id2idx: Dict[int, Dict[int, int]] = defaultdict(dict)

    for facility_id, facility_info in facility_info_dict.items():
        if facility_info.storage_info is not None:
            for i, sku_id in enumerate(facility_info.storage_info.sku_id_list):
                sku_id2idx[facility_id][sku_id] = i

    return sku_id2idx


def get_consumer_id2sku_id(facility_info_dict: Dict[int, FacilityInfo]) -> Dict[int, int]:
    consumer_id2sku_id: Dict[int, int] = {}

    for facility_info in facility_info_dict.values():
        for sku_id, product in facility_info.products_info.items():
            if product.consumer_info:
                consumer_id2sku_id[product.consumer_info.id] = sku_id

    return consumer_id2sku_id


class SCEnvSampler(AbsEnvSampler):
    def __init__(
        self,
        learn_env: Env,
        test_env: Env,
        agent_wrapper_cls: Type[AbsAgentWrapper] = SimpleAgentWrapper,
        reward_eval_delay: Optional[int] = None,
    ) -> None:
        super().__init__(learn_env, test_env, agent_wrapper_cls, reward_eval_delay)

        self._workflow_settings: dict = workflow_settings
        if self._workflow_settings["vehicle_selection_method"] == VehicleSelection.DEFAULT_ONE:
            file_path = os.path.join(
                os.path.dirname(self._learn_env.business_engine._config_path),
                self._learn_env.business_engine._topology,
                "default_vendor.pkl",
            )
            with open(file_path, "rb") as f:
                self._default_vendor = pickle.load(f)

        business_engine = self._learn_env.business_engine
        assert isinstance(business_engine, SupplyChainBusinessEngine)
        self._entity_dict: Dict[int, SupplyChainEntity] = {
            entity.id: entity for entity in business_engine.get_entity_list()
        }

        self._summary: dict = self._learn_env.summary["node_mapping"]

        # Key: Unit id; Value: (unit.data_model_name, unit.data_model_index, unit.facility.id, SkuInfo)
        self._units_mapping: Dict[int, Tuple[str, int, int, SkuInfo]] = self._summary["unit_mapping"]

        self._sku_metas: Dict[int, SkuMeta] = self._summary["skus"]
        self._global_sku_id2idx: Dict[int, int] = {sku_id: idx for idx, sku_id in enumerate(self._sku_metas.keys())}
        self._sku_number: int = len(self._sku_metas)

        self._facility_info_dict: Dict[int, FacilityInfo] = self._summary["facilities"]

        self._unit2product_unit: Dict[int, int] = get_unit2product_unit(self._facility_info_dict)

        # Key 1: Facility id; Key 2: Product id; Value: Index in product list
        self._sku_id2idx_in_product_list: Dict[int, Dict[int, int]] = get_sku_id2idx_in_product_list(
            self._facility_info_dict,
        )

        # Key: Consumer unit id; Value: corresponding product id.
        self._consumer_id2sku_id: Dict[int, int] = get_consumer_id2sku_id(self._facility_info_dict)

        self._configs = self._learn_env.configs  # TODO: Optimize type hint here
        assert isinstance(self._configs, SupplyChainConfiguration)
        self._policy_parameter: Dict[str, Any] = self._parse_policy_parameter(self._configs.policy_parameters)

        self._balance_calculator: BalanceSheetCalculator = BalanceSheetCalculator(self._learn_env, TEAM_REWARD)

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
        self._facility_in_transit_quantity: Dict[int, List[int]] = {}
        self._facility_to_distribute_quantity: Dict[int, List[int]] = {}

        self._facility_product_utilization: Dict[int, int] = {}

        # Key: facility id
        self._storage_product_quantity: Dict[int, List[int]] = defaultdict(lambda: [0] * self._sku_number)

        self._storage_capacity_dict: Optional[Dict[int, Dict[int, int]]] = None

        self._cached_vlt: Dict[int, VendorLeadingTimeInfo] = {}
        self._fixed_vlt = self._workflow_settings["vehicle_selection_method"] != VehicleSelection.RANDOM

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
            settings=self._workflow_settings,
        )

        self.baseline_policy = ConsumerBaselinePolicy("baseline_eoq")

        self.policy_action_by_quantity = [BaseStockPolicy]

        self._or_agent_states: ScOrAgentStates = ScOrAgentStates(
            entity_dict=self._entity_dict,
            facility_info_dict=self._facility_info_dict,
            global_sku_id2idx=self._global_sku_id2idx,
        )

        ########################################################################
        # Evaluation Result Render.
        ########################################################################
        self._is_eval: bool = False

        self._tracker = SimulationTracker(
            episode_len=test_env_conf["durations"],
            n_episodes=1,
            env_sampler=self,
            log_path=self._workflow_settings["log_path"],
            eval_period=[env_conf["durations"], test_env_conf["durations"]],
            # eval_period=[0, test_env_conf["durations"]],
        )

        # Status for evaluation results tracker.
        self._stock_status: Dict[int, int] = {}
        self._demand_status: Dict[int, Union[int, float]] = {}
        self._sold_status: Dict[int, Union[int, float]] = {}
        self._stock_in_transit_status: Dict[int, int] = {}
        self._stock_ordered_to_distribute_status: Dict[int, int] = {}
        self._reward_status: Dict[int, float] = {}
        self._balance_status: Dict[int, float] = {}

        # Evaluation statistics.
        self._eval_reward_list: List[float] = []
        self._max_eval_reward: float = np.float("-inf")

        self._mean_reward: Dict[int, float] = defaultdict(float)
        self._step_idx = 0
        self._eval_reward = 0.0  # only consider those that are associated with RLPolicy

        self.product_metric_track: Dict[str, list] = defaultdict(list)

        self._logger = Logger(
            tag="env_sampler",
            format_=LogFormat.time_only,
            dump_folder=self._workflow_settings["log_path"],
        )

        shutil.copy(
            src=os.path.join(os.path.dirname(__file__), "config.py"),
            dst=self._workflow_settings["log_path"],
        )

    def build(self, rl_component_bundle: RLComponentBundle) -> None:
        super().build(rl_component_bundle)

        self._logger.info(
            f"Total number of policy-related agents / entities: "
            f"{len(self._agent2policy.keys())} / {len(self._entity_dict.keys())}",
        )

    def _parse_policy_parameter(self, raw_info: dict) -> Dict[str, Any]:
        facility_name2id: Dict[str, int] = {
            facility_info.name: facility_id for facility_id, facility_info in self._facility_info_dict.items()
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
            sku_id_list = storage_snapshots[0:node_index:"sku_id_list"].flatten().astype(int)

            for sku_id, sub_storage_idx in zip(sku_id_list, product_storage_index_list):
                storage_capacity_dict[node_index][sku_id] = storage_capacity_list[sub_storage_idx]

        return storage_capacity_dict

    def _get_reward_for_entity(self, entity: SupplyChainEntity, bwt: Tuple[float, float]) -> float:
        if issubclass(entity.class_type, ConsumerUnit):
            return np.float32(bwt[1]) / np.float32(self._workflow_settings["reward_normalization"])
        else:
            return 0.0

    def _get_upstream_facility_id_to_sku_info_dict(self, facility_id: int, sku_id: int) -> Optional[Dict[int, SkuInfo]]:
        sku_id_to_facility_id_list = self._facility_info_dict[facility_id].upstreams
        sku_id_to_facility_id_list: Dict[int, List[int]] = self._facility_info_dict[facility_id].upstreams
        if sku_id in sku_id_to_facility_id_list:
            upstream_facility_id_list = sku_id_to_facility_id_list[sku_id]
            facility_info_list = [self._facility_info_dict[facility_id] for facility_id in upstream_facility_id_list]
            facility_id_to_sku_info = {facility.id: facility_info.skus[sku_id] for facility_info in facility_info_list}
            return facility_id_to_sku_info
        else:
            return None

    def _get_vlt_info(self, entity_id: int) -> Optional[VendorLeadingTimeInfo]:
        if entity_id not in self._cached_vlt:
            entity = self._entity_dict[entity_id]
            assert issubclass(entity.class_type, (ConsumerUnit, ManufactureUnit))
            facility_info: FacilityInfo = self._facility_info_dict[entity.facility_id]
            sku_id: int = entity.skus.id
            sku_name: str = entity.skus.name

            info_by_fid = facility_info.upstream_vlt_infos[sku_id]
            vlt_info_candidates: List[VendorLeadingTimeInfo] = [
                info for info_by_type in info_by_fid.values() for info in info_by_type.values()
            ]

            if len(vlt_info_candidates) > 0:
                vehicle_selection = self._workflow_settings["vehicle_selection_method"]
                if vehicle_selection == VehicleSelection.RANDOM:
                    vlt_info = random.choice(vlt_info_candidates)
                elif vehicle_selection == VehicleSelection.SHORTEST_LEADING_TIME:
                    vlt_info = min(vlt_info_candidates, key=lambda x: x.vlt)
                elif vehicle_selection == VehicleSelection.CHEAPEST_TOTAL_COST:
                    # As the product cost and order base cost are only related to product quantity,
                    # the transportation cost is the difference of different vehicle type selections.
                    vlt_info = min(vlt_info_candidates, key=lambda x: x.unit_transportation_cost * (x.vlt + 1))
                elif vehicle_selection == VehicleSelection.DEFAULT_ONE:
                    # TODO: read default vlt info directly.
                    default_vehicle_type = self._default_vendor[facility_info.name][sku_name]
                    vlt_info = next(
                        filter(lambda info: info.vehicle_type == default_vehicle_type, vlt_info_candidates),
                        None,
                    )
                    assert (
                        vlt_info is not None
                    ), f"Default vehicle type {default_vehicle_type} not exist for {facility_info.name}'s {sku_name}!"
                else:
                    raise Exception(f"Vehicle Selection method undefined: {vehicle_selection}")
            else:
                vlt_info = None

            self._cached_vlt[entity_id] = vlt_info

        return self._cached_vlt[entity_id]

    def dump_vlt_info(self) -> None:
        default_vendor = {}
        for entity_id, vlt_info in self._cached_vlt.items():
            entity = self._entity_dict[entity_id]
            facility_info = self._facility_info_dict[entity.facility_id]
            sku_id = entity.skus.id

            facility_name = facility_info.name
            sku_name = entity.skus.name

            for src_fid, info_by_type in facility_info.upstream_vlt_infos[sku_id].items():
                for v_type, info in info_by_type.items():
                    if info == vlt_info:
                        if facility_name not in default_vendor:
                            default_vendor[facility_name] = {}
                        default_vendor[facility_name][sku_name] = (self._facility_info_dict[src_fid].name, v_type)
                        break

        # assert isinstance(self._workflow_settings["vehicle_selection_method"], VehicleSelection)
        # selection_method = self._workflow_settings["vehicle_selection_method"].value

        file_path = os.path.join(self._workflow_settings["log_path"], f"vendor.py")
        pprint_path = os.path.join(self._workflow_settings["log_path"], f"vendor_pprint.py")
        with open(file_path, "w") as f:
            json.dump(default_vendor, f)

        stdout_fh = sys.stdout
        sys.stdout = open(pprint_path, "w")
        pprint(default_vendor)
        sys.stdout.close()
        sys.stdout = stdout_fh

    def get_or_policy_state(self, entity: SupplyChainEntity) -> dict:
        if self._storage_capacity_dict is None:
            self._storage_capacity_dict = self._get_storage_capacity_dict_info()

        upstream_facility_id_to_sku_info_dict: Dict[int, SkuInfo] = self._get_upstream_facility_id_to_sku_info_dict(
            entity.facility_id,
            entity.skus.id,
        )

        if upstream_facility_id_to_sku_info_dict is not None:
            upstream_prices = [sku.price for sku in upstream_facility_id_to_sku_info_dict.values()]
            upstream_price_mean = np.mean(upstream_prices)
        else:
            upstream_price_mean = None
        state = self._or_agent_states.update_entity_state(
            entity_id=entity.id,
            tick=self._env.tick,
            storage_capacity_dict=self._storage_capacity_dict,
            product_metrics=self._cur_metrics["products"].get(self._unit2product_unit[entity.id], None),
            product_levels=self._storage_product_quantity[entity.facility_id],
            in_transit_quantity=self._facility_in_transit_quantity[entity.facility_id],
            to_distribute_quantity=self._facility_to_distribute_quantity[entity.facility_id],
            upstream_price_mean=upstream_price_mean,
            history_demand=self.history_demand,
            history_price=self.history_price,
            history_purchased=self.history_purchased,
            chosen_vlt_info=self._get_vlt_info(entity.id),
            fixed_vlt=self._fixed_vlt,
            start_date_time=datetime.datetime.strptime(self._env.configs.settings["start_date_time"], "%Y-%m-%d"),
            durations=self._test_env._durations,
        )
        return state

    def get_rl_policy_state(self, entity_id: int) -> np.ndarray:
        state = self._rl_agent_states.update_entity_state(
            entity_id=entity_id,
            tick=self._env.tick,
            cur_metrics=self._cur_metrics,
            cur_distribution_states=self._cur_distribution_states,
            cur_seller_hist_states=self._cur_seller_hist_states,
            cur_consumer_hist_states=self._cur_consumer_hist_states,
            accumulated_balance=self._balance_calculator.accumulated_balance_sheet[entity_id],
            storage_product_quantity=self._storage_product_quantity,
            facility_product_utilization=self._facility_product_utilization,
            facility_in_transit_quantity=self._facility_in_transit_quantity,
            chosen_vlt_info=self._get_vlt_info(entity_id),
            fixed_vlt=self._fixed_vlt,
        )

        entity = self._entity_dict[entity_id]
        assert issubclass(entity.class_type, ConsumerUnit)
        baseline_state = self.get_or_policy_state(entity)
        baseline_action = self.baseline_policy.get_actions([baseline_state])[0]
        state["baseline_action"] = [0] * OR_NUM_CONSUMER_ACTIONS
        state["baseline_action"][baseline_action] = 1.0

        np_state = serialize_state(state)
        return np_state

    def _get_entity_state(self, entity_id: int) -> Union[np.ndarray, dict, None]:
        entity = self._entity_dict[entity_id]
        policy = self._policy_dict[self._agent2policy[entity_id]]

        if isinstance(policy, RLPolicy):
            return self.get_rl_policy_state(entity_id)
        elif isinstance(policy, RuleBasedPolicy):
            return self.get_or_policy_state(entity)
        else:
            return None

    def _update_eval_tracker_status(self) -> None:
        for entity_id in self._agent2policy.keys():  # TODO: Check use agent2policy.keys() or entity_info.keys()
            entity = self._entity_dict[entity_id]
            assert issubclass(entity.class_type, (ConsumerUnit, ManufactureUnit))

            self._stock_status[entity_id] = self._storage_product_quantity[entity.facility_id][
                self._global_sku_id2idx[entity.skus.id]
            ]
            self._stock_in_transit_status[entity_id] = self._facility_in_transit_quantity[entity.facility_id][
                self._global_sku_id2idx[entity.skus.id]
            ]

            pending_order = self._cur_metrics["facilities"][entity.facility_id]["pending_order"]
            self._stock_ordered_to_distribute_status[entity_id] = pending_order[entity.skus.id] if pending_order else 0

            product_unit_id = entity.parent_id
            self._demand_status[entity_id] = self._cur_metrics["products"][product_unit_id]["demand_mean"]
            self._sold_status[entity_id] = self._cur_metrics["products"][product_unit_id]["sale_mean"]

    def _get_global_and_agent_state_impl(
        self,
        event: CascadeEvent,
        tick: int = None,
    ) -> Tuple[Union[None, np.ndarray, List[object]], Dict[Any, Union[np.ndarray, List[object]]]]:
        """Update the status variables first, then call the state shaper for each agent."""
        if tick is None:
            tick = self._env.tick
        else:
            # To make sure the usage of metrics is correct, the tick should be same to the current env tick.
            assert tick == self._env.tick

        self._cur_metrics = self._env.metrics

        # Get distribution features of current tick from snapshot list.
        self._cur_distribution_states = (
            self._env.snapshot_list["distribution"][tick::distribution_features]
            .flatten()
            .reshape(-1, len(distribution_features))
            .astype(np.int)
        )

        # Get consumer features of specific ticks from snapshot list.
        consumption_hist_ticks = [tick - i for i in range(self._workflow_settings["consumption_hist_len"] - 1, -1, -1)]
        self._cur_consumer_hist_states = self._env.snapshot_list["consumer"][
            consumption_hist_ticks::consumer_features
        ].reshape(self._workflow_settings["consumption_hist_len"], -1, len(consumer_features))

        # Get seller features of specific ticks from snapshot list.
        sale_hist_ticks = [tick - i for i in range(self._workflow_settings["sale_hist_len"] - 1, -1, -1)]
        self._cur_seller_hist_states = (
            self._env.snapshot_list["seller"][sale_hist_ticks::seller_features]
            .reshape(self._workflow_settings["sale_hist_len"], -1, len(seller_features))
            .astype(np.int)
        )

        history_feature_shape = (len(self._env.snapshot_list), -1)
        # Get all history demand from snapshot list.
        self.history_demand = self._env.snapshot_list["seller"][:: seller_features[IDX_SELLER_DEMAND]].reshape(
            history_feature_shape,
        )

        # Get all history selling price from snapshot list.
        self.history_price = self._env.snapshot_list["product"][:: product_features[IDX_PRODUCT_PRICE]].reshape(
            history_feature_shape,
        )

        self.history_purchased = self._env.snapshot_list["consumer"][
            :: consumer_features[IDX_CONSUMER_PURCHASED]
        ].reshape(history_feature_shape)

        # 1. Update storage product quantity info.
        # 2. Update facility product utilization info.
        # 3. Update facility in transition order quantity info.
        for facility_id, facility_info in self._facility_info_dict.items():
            # Reset for each step
            self._facility_product_utilization[facility_id] = 0
            self._facility_in_transit_quantity[facility_id] = [0] * self._sku_number
            self._facility_to_distribute_quantity[facility_id] = [0] * self._sku_number
            if facility_info.storage_info.node_index is not None:
                product_quantities = (
                    self._env.snapshot_list["storage"][
                        tick : facility_info.storage_info.node_index : "product_quantity"
                    ]
                    .flatten()
                    .astype(np.int)
                )

                for pid, index in self._sku_id2idx_in_product_list[facility_id].items():
                    product_quantity = product_quantities[index]

                    self._storage_product_quantity[facility_id][self._global_sku_id2idx[pid]] = product_quantity
                    self._facility_product_utilization[facility_id] += product_quantity

            for sku_id, product_info in facility_info.products_info.items():
                if product_info.consumer_info is None:
                    continue
                consumer_index = product_info.consumer_info.node_index
                quantity = self._env.snapshot_list["consumer"][tick:consumer_index:"in_transit_quantity"].flatten()[0]
                self._facility_in_transit_quantity[facility_id][self._global_sku_id2idx[sku_id]] = quantity

            for sku_id, quantity in self._cur_metrics["facilities"][facility_id]["pending_order"].items():
                self._facility_to_distribute_quantity[facility_id][self._global_sku_id2idx[sku_id]] = quantity

        state = {id_: self._get_entity_state(id_) for id_ in self._agent2policy.keys()}

        # NOTE: update tracker status after call get_entity_state to get the updated rl states.
        if self._is_eval:
            self._update_eval_tracker_status()

        return None, state

    def _get_reward(self, env_action_dict: Dict[Any, object], event: object, tick: int) -> Dict[Any, float]:
        # get related product, seller, consumer, manufacture unit id
        # NOTE: this mapping does not contain facility id, so if id is not exist, then means it is a facility
        self._cur_balance_sheet_reward = self._balance_calculator.calc_and_update_balance_sheet(tick=tick)

        if self._is_eval:
            self._balance_status = {
                f_id: np.float32(reward[0]) for f_id, reward in self._cur_balance_sheet_reward.items()
            }
            self._reward_status = {
                f_id: np.float32(reward[1]) for f_id, reward in self._cur_balance_sheet_reward.items()
            }

        rewards = {
            unit_id: self._get_reward_for_entity(self._entity_dict[unit_id], bwt)
            for unit_id, bwt in self._cur_balance_sheet_reward.items()
            if unit_id in self._agent2policy
        }

        def get_reward_norm(entity_id):
            entity = self._entity_dict[entity_id]
            if (not TEAM_REWARD) and issubclass(entity.class_type, ConsumerUnit):
                return entity.skus.price + 1e-3
            else:
                return 1.0

        return {entity_id: r / get_reward_norm(entity_id) for entity_id, r in rewards.items()}

    def _translate_to_env_action(
        self,
        action_dict: Dict[Any, Union[np.ndarray, List[object]]],
        event: object,
    ) -> Dict[Any, object]:
        env_action_dict: Dict[int, SupplyChainAction] = {}

        for agent_id, action in action_dict.items():
            entity_id = agent_id
            env_action: Optional[SupplyChainAction] = None
            if np.isscalar(action):
                action = [action]

            # Consumer action
            if issubclass(self._entity_dict[agent_id].class_type, ConsumerUnit):
                if isinstance(self._policy_dict[self._agent2policy[agent_id]], RLPolicy):
                    baseline_action = np.array(self._agent_state_dict[agent_id][-OR_NUM_CONSUMER_ACTIONS:])
                    or_action = np.where(baseline_action == 1.0)[0][0]
                    action_idx = max(0, int(action[0] - 1 + or_action))
                else:
                    action_idx = action[0]

                product_unit_id: int = self._unit2product_unit[entity_id]
                if type(self._policy_dict[self._agent2policy[agent_id]]) in self.policy_action_by_quantity:
                    action_quantity = action_idx
                else:
                    action_quantity = int(
                        int(action_idx) * max(1.0, self._cur_metrics["products"][product_unit_id]["demand_mean"]),
                    )

                # Ignore 0 quantity to reduce action number
                if action_quantity:
                    sku_id: int = self._consumer_id2sku_id.get(entity_id, 0)
                    vlt_info = self._get_vlt_info(agent_id)
                    env_action = ConsumerAction(
                        id=entity_id,
                        sku_id=sku_id,
                        source_id=vlt_info.src_facility.id,
                        quantity=action_quantity,
                        vehicle_type=vlt_info.vehicle_type,
                    )

            # Manufacture action
            elif issubclass(self._entity_dict[agent_id].class_type, ManufactureUnit):
                if action[0] > 0:
                    env_action = ManufactureAction(id=entity_id, manufacture_rate=action[0])

            if env_action:
                env_action_dict[agent_id] = env_action

        return env_action_dict

    def _reset(self):
        super()._reset()

        if self._is_eval:
            self._mean_reward.clear()
            self._step_idx = 0
            self._eval_reward = 0.0

            self.product_metric_track.clear()

    def _post_step(self, cache_element: CacheElement) -> None:
        if not self._fixed_vlt:
            self._cached_vlt.clear()

    def post_collect(self, info_list: list, ep: int) -> None:
        return super().post_collect(info_list, ep)

    def sample(self, policy_state: Optional[Dict[str, object]] = None, num_steps: Optional[int] = None) -> dict:
        self._is_eval = False
        self._balance_calculator.update_env(self._learn_env)
        return super().sample(policy_state, num_steps)

    def _step_product_metric_track(self, tick) -> None:
        self._cur_metrics = self._env._business_engine.get_metrics()

        for facility_id, facility_info in self._facility_info_dict.items():
            for product_info in facility_info.products_info.values():
                self.product_metric_track["tick"].append(tick)

                # TODO: it could be got from snapshot list.
                self.product_metric_track["inventory_in_transit"].append(
                    self._cur_metrics["facilities"][facility_id]["in_transit_orders"][product_info.sku_id],
                )

                pending_orders = self._cur_metrics["facilities"][facility_id]["pending_order"]
                self.product_metric_track["inventory_to_distribute"].append(
                    pending_orders[product_info.sku_id] if pending_orders else 0,
                )

    def _post_update_product_metric_track(self) -> None:
        static_product_metrics = defaultdict(list)
        for facility_id, facility_info in self._facility_info_dict.items():
            distribution_info: DistributionUnitInfo = facility_info.distribution_info
            storage_info: StorageUnitInfo = facility_info.storage_info

            for product_info in facility_info.products_info.values():
                static_product_metrics["id"].append(product_info.id)
                static_product_metrics["sku_id"].append(product_info.sku_id)
                static_product_metrics["facility_id"].append(facility_id)
                static_product_metrics["facility_name"].append(facility_info.name)
                static_product_metrics["name"].append(self._sku_metas[product_info.sku_id].name)
                static_product_metrics["unit_inventory_holding_cost"].append(
                    self._entity_dict[product_info.id].skus.unit_storage_cost,
                )

                # The indexes below are only used for accessing dynamic metrics
                static_product_metrics["product_node_index"].append(product_info.node_index)
                static_product_metrics["consumer_node_index"].append(
                    product_info.consumer_info.node_index if product_info.consumer_info else None,
                )
                static_product_metrics["seller_node_index"].append(
                    product_info.seller_info.node_index if product_info.seller_info else None,
                )
                static_product_metrics["manufacture_node_index"].append(
                    product_info.manufacture_info.node_index if product_info.manufacture_info else None,
                )
                static_product_metrics["distribution_node_index"].append(
                    distribution_info.node_index if distribution_info else None,
                )
                static_product_metrics["storage_node_index"].append(storage_info.node_index if storage_info else None)

        num_ticks = self.product_metric_track["tick"][-1] - self.product_metric_track["tick"][0] + 1
        for key, value_list in static_product_metrics.items():
            self.product_metric_track[key] = value_list * num_ticks

        for (name, features) in [
            ("product", ("price", "check_in_quantity_in_order", "delay_order_penalty", "transportation_cost")),
            ("consumer", ("purchased", "received", "order_base_cost", "order_product_cost")),
            ("seller", ("sold", "demand", "backlog_ratio")),
            (
                "manufacture",
                ("finished_quantity", "in_pipeline_quantity", "manufacture_cost", "start_manufacture_quantity"),
            ),
            ("distribution", ("pending_product_quantity", "pending_order_number")),
        ]:
            for feature in features:
                value_dict = get_attributes(self._env, name, feature, tick=None)
                self.product_metric_track[f"{name}_{feature}"] = [
                    value_dict[tick][node_index] if node_index else 0
                    for tick, node_index in zip(
                        self.product_metric_track["tick"],
                        self.product_metric_track[f"{name}_node_index"],
                    )
                ]

        sku_id_lists = get_list_attributes(self._env, "storage", "sku_id_list", tick=0)
        quantity_lists = get_list_attributes(self._env, "storage", "product_quantity", tick=None)

        quantity_dict = {}
        for tick in self.product_metric_track["tick"]:
            if tick in quantity_dict:
                continue
            quantity_dict[tick] = {}

            for node_index in self.product_metric_track["storage_node_index"]:
                if node_index is None or node_index in quantity_dict[tick]:
                    continue

                quantity_dict[tick][node_index] = {
                    sku_id: quantity
                    for sku_id, quantity in zip(
                        sku_id_lists[node_index].astype(np.int),
                        quantity_lists[tick][node_index].astype(np.int),
                    )
                }

        self.product_metric_track["inventory_in_stock"] = [
            quantity_dict[tick][node_index][sku_id] if node_index else 0
            for tick, node_index, sku_id in zip(
                self.product_metric_track["tick"],
                self.product_metric_track[f"storage_node_index"],
                self.product_metric_track["sku_id"],
            )
        ]

        for key in [
            "product_node_index",
            "consumer_node_index",
            "seller_node_index",
            "manufacture_node_index",
            "distribution_node_index",
            "storage_node_index",
        ]:
            self.product_metric_track.pop(key)

    def _post_eval_step(self, cache_element: CacheElement) -> None:
        self._logger.info(f"Step: {self._step_idx}")
        if self._tracker.eval_period[0] <= cache_element.tick < self._tracker.eval_period[1]:
            self._eval_reward += np.sum(
                [
                    self._balance_status[entity_id]
                    for entity_id, entity in self._entity_dict.items()
                    if issubclass(entity.class_type, StoreProductUnit)
                ],
            )

        if self._workflow_settings["log_consumer_actions"]:
            consumer_action_dict = {}
            for entity_id in cache_element.agent_state_dict.keys():
                entity = self._entity_dict[entity_id]
                self._mean_reward[entity_id] += self._reward_status.get(entity_id, 0)
                if issubclass(entity.class_type, ConsumerUnit):
                    parent_entity = self._entity_dict[entity.parent_id]
                    if issubclass(parent_entity.class_type, StoreProductUnit):
                        action = (
                            cache_element.action_dict[entity_id]
                            if np.isscalar(cache_element.action_dict[entity_id])
                            else cache_element.action_dict[entity_id][0]
                        )
                        or_action = 0
                        if ALGO != "EOQ":
                            baseline_action = np.array(
                                cache_element.agent_state_dict[entity_id][-OR_NUM_CONSUMER_ACTIONS:],
                            )
                            or_action = np.where(baseline_action == 1.0)[0][0]
                        consumer_action_dict[parent_entity.id] = (
                            action,
                            or_action,
                            round(cache_element.reward_dict[entity_id], 2),
                        )
            # self._logger.debug(f"Consumer_action_dict: {consumer_action_dict}")

        self._tracker.add_balance_and_reward(
            episode=0,
            tick=self._step_idx,
            global_balance=sum(self._balance_status.values()),
            global_reward=sum(self._reward_status.values()),
            step_balances=self._balance_status,
            step_rewards=self._reward_status,
        )

        self._tracker.add_sku_status(
            episode=0,
            tick=self._step_idx,
            stock=self._stock_status,
            stock_in_transit=self._stock_in_transit_status,
            stock_ordered_to_distribute=self._stock_ordered_to_distribute_status,
            demands=self._demand_status,
            solds=self._sold_status,
            rewards=self._reward_status,
            balances=self._balance_status,
        )

        self._step_idx += 1

        # self._logger.info(f"tracker sample & sku status added")

        if self._workflow_settings["dump_product_metrics"]:
            self._step_product_metric_track(cache_element.tick)
            # self._logger.info(f"dump step product metrics updated")

        if not self._fixed_vlt:
            self._cached_vlt.clear()

    def _get_action_status(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        first_tick, last_tick = self._tracker.eval_period
        len_period = last_tick - first_tick
        status_shape = (1, len_period, len(self._tracker.tracking_entity_ids))
        consumer_purchased = np.zeros(status_shape)
        consumer_received = np.zeros(status_shape)
        manufacture_started = np.zeros(status_shape)
        manufacture_finished = np.zeros(status_shape)

        ticks = list(range(first_tick, last_tick))
        purchased = self._env.snapshot_list["consumer"][ticks::"purchased"].reshape(len_period, -1)
        received = self._env.snapshot_list["consumer"][ticks::"received"].reshape(len_period, -1)
        started = self._env.snapshot_list["manufacture"][ticks::"start_manufacture_quantity"].reshape(len_period, -1)
        finished = self._env.snapshot_list["manufacture"][ticks::"finished_quantity"].reshape(len_period, -1)

        for i, entity_id in enumerate(self._tracker.tracking_entity_ids):
            entity = self._entity_dict[entity_id]

            if issubclass(entity.class_type, FacilityBase):
                continue

            if issubclass(entity.class_type, (ConsumerUnit, ProductUnit)):
                product_info = self._facility_info_dict[entity.facility_id].products_info[entity.skus.id]
                if product_info.consumer_info is not None:
                    node_idx = product_info.consumer_info.node_index
                    consumer_purchased[0, :, i] = purchased[:, node_idx]
                    consumer_received[0, :, i] = received[:, node_idx]

            if issubclass(entity.class_type, (ManufactureUnit, ProductUnit)):
                product_info = self._facility_info_dict[entity.facility_id].products_info[entity.skus.id]
                if product_info.manufacture_info is not None:
                    node_idx = product_info.manufacture_info.node_index
                    manufacture_started[0, :, i] = started[:, node_idx]
                    manufacture_finished[0, :, i] = finished[:, node_idx]

        return consumer_purchased, consumer_received, manufacture_started, manufacture_finished

    def post_evaluate(self, info_list: list, ep: int) -> None:
        self._eval_reward_list.append(self._eval_reward)

        consumer_purchased, consumer_received, manufacture_started, manufacture_finished = self._get_action_status()
        self._tracker.add_action_status(
            consumer_purchased=consumer_purchased,
            consumer_received=consumer_received,
            manufacture_started=manufacture_started,
            manufacture_finished=manufacture_finished,
        )

        if self._eval_reward > self._max_eval_reward:
            self._max_eval_reward = self._eval_reward
            self._logger.info(f"Update Max Eval Reward to: {self._max_eval_reward:,.2f}")

            if self._workflow_settings["plot_render"]:
                self._logger.info("Start render...")
                self._tracker.render_facility_balance_and_reward(facility_types=(OuterRetailerFacility))
                self._tracker.render_all_sku(entity_types=(ConsumerUnit, ManufactureUnit))
            self._tracker.dump_sku_status(entity_types=(ConsumerUnit, ManufactureUnit))

            if self._workflow_settings["dump_product_metrics"]:
                self._logger.info(f"Start post update product metrics...")
                self._post_update_product_metric_track()

                self._logger.info("Start dump product metrics...")
                df_product = pd.DataFrame(self.product_metric_track)
                df_product = df_product.groupby(["tick", "id"]).first().reset_index()
                df_product.to_csv(
                    os.path.join(self._workflow_settings["log_path"], "output_product_metrics.csv"),
                    index=False,
                )

                self._logger.info("product metrics dumped to csv")

        self._logger.info(f"Max Eval Reward: {self._max_eval_reward:,.2f}")
        self._logger.debug(f"Eval Reward List: {self._eval_reward_list}")
        self._mean_reward = {entity_id: val / self._step_idx for entity_id, val in self._mean_reward.items()}

        if self._workflow_settings["dump_chosen_vlt_info"]:
            self.dump_vlt_info()

    def eval(self, policy_state: Dict[str, object] = None) -> dict:
        self._is_eval = True
        self._balance_calculator.update_env(self._test_env)
        return super().eval(policy_state)
