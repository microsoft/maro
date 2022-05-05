# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import collections
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple
from maro.simulator.scenarios.supply_chain.units.product import StoreProductUnit

import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain.facilities.facility import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity
from maro.simulator.scenarios.supply_chain.units.distribution import DistributionUnitInfo
from maro.simulator.scenarios.supply_chain.units.storage import StorageUnitInfo
from maro.simulator.scenarios.supply_chain import RetailerFacility
# from ..rl.env_helper import STORAGE_INFO


ProductInfo = namedtuple(
    "ProductInfo",
    (
        "unit_id",
        "sku_id",
        "node_index",
        "facility_id",
        "storage_index",
        "distribution_index",
        "downstream_product_unit_id_list",
        "product_index",
        "consumer_index",
        "seller_index",
        "manufacture_index",
    ),
)


FacilityLevelInfo = namedtuple(
    "FacilityLevelInfo",
    (
        "unit_id",
        "product_unit_id_list",
        "storage_index",
        "distribution_index",
    ),
)


class BalanceSheetCalculator:
    consumer_features = ("id", "purchased", "received",
                         "order_base_cost", "order_product_cost")
    seller_features = ("id", "sold", "demand", "backlog_ratio")
    manufacture_features = ("id", "finished_quantity", 'in_pipeline_quantity', 'manufacture_cost', 'start_manufacture_quantity')
    product_features = (
        "id", "price", 'check_in_quantity_in_order', 'delay_order_penalty', "transportation_cost")
    storage_features = ("capacity", "remaining_space")
    distribution_features = ("pending_product_quantity", "pending_order_number")

    def __init__(self, env: Env, team_reward) -> None:
        self._env: Env = env
        self._team_reward = team_reward

        self._facility_info_dict: Dict[int, FacilityInfo] = self._env.summary["node_mapping"]["facilities"]

        self._entity_dict: Dict[int, SupplyChainEntity] = {
            entity.id: entity
            for entity in self._env.business_engine.get_entity_list()
        }

        facilities, products, pid2idx, cid2pid, sidx2pidx = self._extract_facility_and_product_info()

        self.facility_levels: List[FacilityLevelInfo] = facilities
        self.products: List[ProductInfo] = products

        self.product_id2idx: Dict[int, int] = pid2idx
        self.consumer_id2product_id: Dict[int, int] = cid2pid

        self.seller_idx2product_idx: Dict[int, int] = sidx2pidx

        self.num_products = len(self.products)
        self.num_facilities = len(self.facility_levels)

        self._ordered_products: List[ProductInfo] = self._get_products_sorted_from_downstreams_to_upstreams()

        self.accumulated_balance_sheet = defaultdict(int)
        self.product_metric_track = {}
        self.tick_cached = set()
        self.facility_info = self._env.summary['node_mapping']['facilities']
        self.sku_meta_info = self._env.summary['node_mapping']['skus']

        for col in ['id', 'sku_id', 'facility_id', 'facility_name', 'name', 'tick', 'inventory_in_stock', 
                    'inventory_in_transit', 'inventory_to_distribute', 'unit_inventory_holding_cost', "seller_price"]:
            self.product_metric_track[col] = []
        for (name, cols) in [("consumer", self.consumer_features),
                             ("seller", self.seller_features),
                             ("manufacture", self.manufacture_features),
                             ("product", self.product_features),
                             ("distribution", self.distribution_features)]:
            for fea in cols:
                if fea == 'id':
                    continue
                self.product_metric_track[f"{name}_{fea}"] = []

    def _extract_facility_and_product_info(self) -> Tuple[
        List[FacilityLevelInfo], List[ProductInfo], Dict[int, int], Dict[int, int],
    ]:
        facility_levels: List[FacilityLevelInfo] = []
        products: List[ProductInfo] = []

        product_id2idx: Dict[int, int] = {}
        consumer_id2product_id: Dict[int, int] = {}

        seller_idx2product_idx: Dict[int, int] = {}

        for facility_id, facility_info in self._facility_info_dict.items():
            distribution_info: DistributionUnitInfo = facility_info.distribution_info
            storage_info: StorageUnitInfo = facility_info.storage_info
            downstreams: Dict[int, List[int]] = facility_info.downstreams

            product_id_list = []
            for product_id, product_info in facility_info.products_info.items():
                product_id_list.append(product_info.id)
                product_id2idx[product_info.id] = len(products)

                if product_info.consumer_info:
                    consumer_id2product_id[product_info.consumer_info.id] = product_info.id

                if product_info.seller_info:
                    seller_idx2product_idx[product_info.seller_info.node_index] = product_info.node_index

                products.append(
                    ProductInfo(
                        unit_id=product_info.id,
                        sku_id=product_info.product_id,
                        node_index=product_info.node_index,
                        facility_id = facility_id,
                        storage_index=storage_info.node_index,
                        distribution_index=distribution_info.node_index if distribution_info else None,
                        downstream_product_unit_id_list=[
                            self._facility_info_dict[fid].products_info[product_id].id
                            for fid in downstreams[product_id]
                        ] if product_id in downstreams else [],
                        product_index=product_info.node_index,
                        consumer_index=product_info.consumer_info.node_index if product_info.consumer_info else None,
                        seller_index=product_info.seller_info.node_index if product_info.seller_info else None,
                        manufacture_index=(
                            product_info.manufacture_info.node_index if product_info.manufacture_info else None
                        ),
                    )
                )

            facility_levels.append(
                FacilityLevelInfo(
                    unit_id=facility_id,
                    product_unit_id_list=product_id_list,
                    storage_index=storage_info.node_index,
                    distribution_index=distribution_info.node_index if distribution_info else None,
                )
            )

        return facility_levels, products, product_id2idx, consumer_id2product_id, seller_idx2product_idx

    def _get_products_sorted_from_downstreams_to_upstreams(self) -> List[ProductInfo]:
        # Topological sorting
        ordered_products: List[ProductInfo] = []
        product_unit_dict = {product.unit_id: product for product in self.products}
        # in_degree = collections.Counter()

        for product in self.products:
            ordered_products.append(product)
        #     for downstream_unit_id in product.downstream_product_unit_id_list:
        #         in_degree[downstream_unit_id] += 1

        # queue = collections.deque()
        # for unit_id, deg in in_degree.items():
        #     if deg == 0:
        #         queue.append(unit_id)

        # while queue:
        #     unit_id = queue.popleft()
        #     product = product_unit_dict[unit_id]
        #     ordered_products.append(product)
        #     for downstream_unit_id in product.downstream_product_unit_id_list:
        #         in_degree[downstream_unit_id] -= 1
        #         if in_degree[downstream_unit_id] == 0:
        #             queue.append(downstream_unit_id)

        return ordered_products

    def _check_attribute_keys(self, target_type: str, attribute: str) -> None:
        valid_target_types = set(self._env.summary["node_detail"].keys())
        assert target_type in valid_target_types, f"Target_type {target_type} not in {list(valid_target_types)}!"

        valid_attributes = set(self._env.summary["node_detail"][target_type]["attributes"].keys())
        assert attribute in valid_attributes, (
            f"Attribute {attribute} not valid for {target_type}. Valid attributes: {list(valid_attributes)}"
        )

    def _get_attributes(self, target_type: str, attribute: str, tick: int = None) -> np.ndarray:
        self._check_attribute_keys(target_type, attribute)

        if tick is None:
            tick = self._env.tick

        frame_index = self._env.business_engine.frame_index(tick)

        return self._env.snapshot_list[target_type][frame_index::attribute].flatten()

    def _get_list_attributes(self, target_type: str, attribute: str, tick: int = None) -> List[np.ndarray]:
        self._check_attribute_keys(target_type, attribute)

        if tick is None:
            tick = self._env.tick

        frame_index = self._env.business_engine.frame_index(tick)

        indexes = list(range(len(self._env.snapshot_list[target_type])))

        return [self._env.snapshot_list[target_type][frame_index:index:attribute].flatten() for index in indexes]

    def _calc_consumer(self, tick: int) -> Tuple[np.ndarray, np.ndarray]:
        consumer_ids = self._get_attributes("consumer", "id", tick).astype(np.int)

        # order_base_cost + order_product_cost
        consumer_step_cost = -1 * (
            self._get_attributes("consumer", "order_base_cost", tick)
            + self._get_attributes("consumer", "order_product_cost", tick)
        )

        return consumer_ids, consumer_step_cost

    def _calc_seller(self, tick: int) -> Tuple[np.ndarray, np.ndarray]:
        price = self._env.snapshot_list["product"][
            self._env.business_engine.frame_index(tick):[
                self.seller_idx2product_idx[sidx] for sidx in range(len(self._env.snapshot_list["seller"]))
            ]:"price"
        ].flatten()

        # profit = sold * price
        seller_step_profit = self._get_attributes("seller", "sold", tick) * price

        # loss = demand * price * backlog_ratio
        seller_step_cost = -1 * (
            (self._get_attributes("seller", "demand", tick) - self._get_attributes("seller", "sold", tick))
            * self._get_attributes("seller", "backlog_ratio", tick)
            * price
        )

        return seller_step_profit, seller_step_cost

    def _calc_manufacture(self, tick: int) -> Tuple[np.ndarray, np.ndarray]:
        manufacture_ids = self._get_attributes("manufacture", "id", tick).astype(np.int)

        # loss = manufacture number * cost
        manufacture_step_cost = -1 * self._get_attributes("manufacture", "manufacture_cost", tick)

        return manufacture_ids, manufacture_step_cost

# storage_snapshots = self._env.snapshot_list["storage"]
#         for node_index in range(len(storage_snapshots)):
#             storage_capacity_list = storage_snapshots[0:node_index:"capacity"].flatten().astype(int)
#             product_storage_index_list = storage_snapshots[0:node_index:"product_storage_index"].flatten().astype(int)
#             product_id_list = storage_snapshots[0:node_index:"product_list"].flatten().astype(int)

#             for product_id, sub_storage_idx in zip(product_id_list, product_storage_index_list):
#                 storage_capacity_dict[node_index][product_id] = storage_capacity_list[sub_storage_idx]

    def _calc_product_distribution(self, tick: int) -> Tuple[np.ndarray, np.ndarray]:
        # product distribution profit = check order * price
        product_distribution_step_profit = (
            self._get_attributes("product", "check_in_quantity_in_order", tick)
            * self._get_attributes("product", "price", tick)
        )

        # product distribution loss = transportation cost + delay order penalty
        product_distribution_step_cost = -1 * (
            self._get_attributes("product", "transportation_cost", tick)
            + self._get_attributes("product", "delay_order_penalty", tick)
        )

        return product_distribution_step_profit, product_distribution_step_cost

    def _calc_product(
        self,
        consumer_step_cost: np.ndarray,
        manufacture_step_cost: np.ndarray,
        seller_step_profit: np.ndarray,
        seller_step_cost: np.ndarray,
        product_distribution_step_profit: np.ndarray,
        product_distribution_step_cost: np.ndarray,
        tick: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        product_step_profit = np.zeros(self.num_products)
        product_step_cost = np.zeros(self.num_products)

        self._cur_metrics = self._env._business_engine.get_metrics()
        product_price_list = self._env.snapshot_list["product"][
            self._env.business_engine.frame_index(tick)::"price"
        ].flatten()

        # price = self._env.snapshot_list["product"][
        #     self._env.business_engine.frame_index(tick):[
        #         self.seller_idx2product_idx[sidx] for sidx in range(len(self._env.snapshot_list["seller"]))
        #     ]:"price"
        # ].flatten()


        # product = consumer + seller + manufacture + storage + distribution + downstreams
        for i, product in enumerate(self.products):
            node_idx = product.node_index
            storage_index = product.storage_index
            product_storage_index = int(np.where(self._get_list_attributes("storage", "product_list", tick)[storage_index] == product.sku_id)[0])
            stock = int(self._get_list_attributes("storage", "product_quantity", tick)[storage_index][product_storage_index])
            unit_inventory_holding_cost = float(self._entity_dict[product.unit_id].skus.unit_storage_cost)
            if (tick, i) not in self.tick_cached:
                self.tick_cached.add((tick, i))
                meta_sku = self.sku_meta_info[product.sku_id]
                meta_facility = self.facility_info[product.facility_id]
                self.product_metric_track['id'].append(product.unit_id)
                self.product_metric_track['sku_id'].append(product.sku_id)
                self.product_metric_track['tick'].append(tick)
                self.product_metric_track['facility_id'].append(product.facility_id)
                self.product_metric_track['facility_name'].append(meta_facility.name)
                self.product_metric_track['name'].append(meta_sku.name)
                for f_id, fea in enumerate(self.product_features):
                    if fea != 'id':
                        val = self._get_attributes("product", fea, tick)[node_idx]
                        self.product_metric_track[f"product_{fea}"].append(val)
                for fea in self.consumer_features:
                    if fea != 'id':
                        val = (self._get_attributes("consumer", fea, tick)[product.consumer_index] if product.consumer_index is not None else 0)
                        self.product_metric_track[f"consumer_{fea}"].append(val)
                for fea in self.seller_features:
                    if fea != 'id':
                        val = (self._get_attributes("seller", fea, tick)[product.seller_index] if product.seller_index is not None else 0)
                        self.product_metric_track[f"seller_{fea}"].append(val)
                for fea in self.manufacture_features:
                    if fea != 'id':
                        val = (self._get_attributes("manufacture", fea, tick)[product.manufacture_index] if product.manufacture_index is not None else 0)
                        self.product_metric_track[f"manufacture_{fea}"].append(val)
                for fea in self.distribution_features:
                    if fea != 'id':
                        val = (self._get_attributes("distribution", fea, tick)[product.distribution_index] if product.distribution_index is not None else 0)
                        self.product_metric_track[f"distribution_{fea}"].append(val)
                
                self.product_metric_track['unit_inventory_holding_cost'].append(unit_inventory_holding_cost)
                self.product_metric_track['inventory_in_stock'].append(stock)

                in_transit_stock = self._cur_metrics['facilities'][product.facility_id]["in_transit_orders"][product.sku_id]
                self.product_metric_track['inventory_in_transit'].append(in_transit_stock)
                
                if self._cur_metrics['facilities'][product.facility_id]["pending_order"]:      
                    to_distribute_stock = self._cur_metrics['facilities'][product.facility_id]["pending_order"][product.sku_id]
                else:
                    to_distribute_stock = 0
                self.product_metric_track['inventory_to_distribute'].append(to_distribute_stock)
                self.product_metric_track['seller_price'].append(product_price_list[node_idx])

                
            if product.consumer_index:
                product_step_cost[i] += consumer_step_cost[product.consumer_index]

            if product.seller_index:
                product_step_profit[i] += seller_step_profit[product.seller_index]
                product_step_cost[i] += seller_step_cost[product.seller_index]

            if product.manufacture_index:
                product_step_cost[i] += manufacture_step_cost[product.manufacture_index]

            product_step_cost[i] += (-unit_inventory_holding_cost*stock)

            if product.distribution_index:
                product_step_profit[i] += product_distribution_step_profit[i]
                product_step_cost[i] += product_distribution_step_cost[i]

            # TODO: Check if we still need to consider the downstream profit & cost.
            # for did in product.downstream_product_unit_id_list:
            #     product_step_profit[i] += product_step_profit[self.product_id2idx[did]]
            #     product_step_cost[i] += product_step_cost[self.product_id2idx[did]]

        product_step_balance = product_step_profit + product_step_cost

        return product_step_profit, product_step_cost, product_step_balance

    def _calc_facility(
        self,
        storage_step_cost: np.ndarray,
        vehicle_step_cost: np.ndarray,
        product_step_profit: np.ndarray,
        product_step_cost: np.ndarray,
    ) -> tuple:
        facility_step_profit = np.zeros(self.num_facilities)
        facility_step_cost = np.zeros(self.num_facilities)

        for i, facility in enumerate(self.facility_levels):
            # TODO: check is it still needed, since we already add it into the product
            # facility_step_cost[i] += storage_step_cost[facility.storage_index]

            for product_id in facility.product_unit_id_list:
                facility_step_profit[i] += product_step_profit[self.product_id2idx[product_id]]
                facility_step_cost[i] += product_step_cost[self.product_id2idx[product_id]]

        facility_step_balance = facility_step_profit + facility_step_cost

        return facility_step_profit, facility_step_cost, facility_step_balance

    def _update_balance_sheet(
        self,
        product_step_balance: np.ndarray,
        consumer_ids: np.ndarray,
        manufacture_ids: np.ndarray,
        manufacture_step_cost: np.ndarray,
        facility_step_balance: np.ndarray,
    ) -> Dict[int, Tuple[float, float]]:

        # Key: the facility/unit id; Value: (balance, reward).
        balance_and_reward: Dict[int, Tuple[float, float]] = {}

        # For Product Units. TODO: check the logic of reward computation.
        for product, balance in zip(self.products, product_step_balance):
            balance_and_reward[product.unit_id] = (balance, balance)
            self.accumulated_balance_sheet[product.unit_id] += balance

        # For Consumer Units. TODO: check the definitions.
        for id_ in consumer_ids:
            balance_and_reward[id_] = balance_and_reward[self.consumer_id2product_id[id_]]
            self.accumulated_balance_sheet[id_] += balance_and_reward[id_][0]

        # For Manufacture Units. TODO: check the definitions.
        for id_, cost in zip(manufacture_ids, manufacture_step_cost):
            balance_and_reward[id_] = (cost, cost)
            self.accumulated_balance_sheet[id_] += cost

        for cost, facility in zip(facility_step_balance, self.facility_levels):
            id_ = facility.unit_id
            balance_and_reward[id_] = (cost, cost)
            self.accumulated_balance_sheet[id_] += cost

        # team reward
        if self._team_reward:
            team_reward = {}
            for id_ in consumer_ids:
                facility_id = self.products[self.product_id2idx[self.consumer_id2product_id[id_]]].facility_id
                (b, r) = (team_reward[facility_id] if facility_id in team_reward else (0, 0))
                team_reward[facility_id] = (b+balance_and_reward[id_][0], r+balance_and_reward[id_][1])
            for id_ in consumer_ids:
                facility_id = self.products[self.product_id2idx[self.consumer_id2product_id[id_]]].facility_id
                balance_and_reward[id_] = team_reward[facility_id]

        # NOTE: Add followings if needed.
        # For storages.
        # For distributions.
        # For vehicles.

        return balance_and_reward

    def calc_and_update_balance_sheet(self, tick: int) -> Dict[int, Tuple[float, float]]:
        # TODO: Add cache for each tick.
        # TODO: Add logic to confirm the balance sheet for the same tick would not be re-calculate.

        # Basic Units: profit & cost
        consumer_ids, consumer_step_cost = self._calc_consumer(tick)
        seller_step_profit, seller_step_cost = self._calc_seller(tick)
        manufacture_ids, manufacture_step_cost = self._calc_manufacture(tick)
        product_distribution_step_profit, product_distribution_step_cost = self._calc_product_distribution(tick)

        # Product: profit, cost & balance
        product_step_profit, product_step_cost, product_step_balance = self._calc_product(
            consumer_step_cost,
            manufacture_step_cost,
            seller_step_profit,
            seller_step_cost,
            product_distribution_step_profit,
            product_distribution_step_cost,
            tick
        )

        _, _, facility_step_balance = self._calc_facility(None, None,
            product_step_profit, product_step_cost
        )

        balance_and_reward = self._update_balance_sheet(
            product_step_balance, consumer_ids, manufacture_ids, manufacture_step_cost, facility_step_balance
        )

        return balance_and_reward

    def reset(self):
        for key in self.product_metric_track.keys():
            self.product_metric_track[key] = []
        self.tick_cached.clear()