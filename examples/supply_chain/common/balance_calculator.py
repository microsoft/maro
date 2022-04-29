# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import collections
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple

import numpy as np

from maro.simulator import Env

from maro.simulator.scenarios.supply_chain.facilities.facility import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity
from maro.simulator.scenarios.supply_chain.units.distribution import DistributionUnitInfo
from maro.simulator.scenarios.supply_chain.units.storage import StorageUnitInfo


ProductInfo = namedtuple(
    "ProductInfo",
    (
        "unit_id",
        "sku_id",
        "node_index",
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
    def __init__(self, env: Env) -> None:
        self._env: Env = env

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
                product_id2idx[product_id] = len(products)

                if product_info.consumer_info:
                    consumer_id2product_id[product_info.consumer_info.id] = product_info.id

                if product_info.seller_info:
                    seller_idx2product_idx[product_info.seller_info.node_index] = product_info.node_index

                products.append(
                    ProductInfo(
                        unit_id=product_info.id,
                        sku_id=product_info.product_id,
                        node_index=product_info.node_index,
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
        in_degree = collections.Counter()

        for product in self.products:
            for downstream_unit_id in product.downstream_product_unit_id_list:
                in_degree[downstream_unit_id] += 1

        queue = collections.deque()
        for unit_id, deg in in_degree.items():
            if deg == 0:
                queue.append(unit_id)

        while queue:
            unit_id = queue.popleft()
            product = product_unit_dict[unit_id]
            ordered_products.append(product)
            for downstream_unit_id in product.downstream_product_unit_id_list:
                in_degree[downstream_unit_id] -= 1
                if in_degree[downstream_unit_id] == 0:
                    queue.append(downstream_unit_id)

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

    def _calc_storage(self, tick: int) -> List[Dict[int, float]]:
        storage_product_step_cost: List[Dict[int, float]] = [
            {
                product_id: -quantity * self._entity_dict[
                    self.products[self.product_id2idx[product_id]].unit_id
                ].skus.unit_storage_cost
                for product_id, quantity in zip(id_list, quantity_list)
            }
            for id_list, quantity_list in zip(
                [il.astype(np.int) for il in self._get_list_attributes("storage", "product_list", tick)],
                [ql.astype(np.int) for ql in self._get_list_attributes("storage", "product_quantity", tick)],
            )
        ]

        return storage_product_step_cost

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
        storage_product_step_cost: List[Dict[int, float]],
        product_distribution_step_profit: np.ndarray,
        product_distribution_step_cost: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        product_step_profit = np.zeros(self.num_products)
        product_step_cost = np.zeros(self.num_products)

        # product = consumer + seller + manufacture + storage + distribution + downstreams
        for product in self._ordered_products:
            i = product.node_index

            if product.consumer_index:
                product_step_cost[i] += consumer_step_cost[product.consumer_index]

            if product.seller_index:
                product_step_profit[i] += seller_step_profit[product.seller_index]
                product_step_cost[i] += seller_step_cost[product.seller_index]

            if product.manufacture_index:
                product_step_cost[i] += manufacture_step_cost[product.manufacture_index]

            product_step_cost[i] += storage_product_step_cost[product.storage_index][product.sku_id]

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

            for pid in facility.product_unit_id_list:
                facility_step_profit[i] += product_step_profit[self.product_id2idx[pid]]
                facility_step_cost[i] += product_step_cost[self.product_id2idx[pid]]

        facility_step_balance = facility_step_profit + facility_step_cost

        return facility_step_profit, facility_step_cost, facility_step_balance

    def _update_balance_sheet(
        self,
        product_step_balance: np.ndarray,
        consumer_ids: np.ndarray,
        manufacture_ids: np.ndarray,
        manufacture_step_cost: np.ndarray,
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
        storage_product_step_cost = self._calc_storage(tick)
        product_distribution_step_profit, product_distribution_step_cost = self._calc_product_distribution(tick)

        # Product: profit, cost & balance
        product_step_profit, product_step_cost, product_step_balance = self._calc_product(
            consumer_step_cost,
            manufacture_step_cost,
            seller_step_profit,
            seller_step_cost,
            storage_product_step_cost,
            product_distribution_step_profit,
            product_distribution_step_cost,
        )

        balance_and_reward = self._update_balance_sheet(
            product_step_balance, consumer_ids, manufacture_ids, manufacture_step_cost,
        )

        return balance_and_reward
