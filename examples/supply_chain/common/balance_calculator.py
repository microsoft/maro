# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, deque, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain.facilities.facility import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity
from maro.simulator.scenarios.supply_chain.units.distribution import DistributionUnitInfo
from maro.simulator.scenarios.supply_chain.units.storage import StorageUnitInfo

from .utils import get_attributes, get_list_attributes


@dataclass
class GlobalProductInfo:
    unit_id: int
    sku_id: int
    node_index: int
    storage_index: Optional[int]
    distribution_index: Optional[int]
    consumer_index: Optional[int]
    seller_index: Optional[int]
    manufacture_index: Optional[int]
    downstream_product_id_list: List[int]
    upstream_product_id_list: List[int]


class BalanceSheetCalculator:
    def __init__(self, env: Env, team_reward: bool) -> None:
        self._env: Env = env
        self._team_reward: bool = team_reward

        self._facility_info_dict: Dict[int, FacilityInfo] = self._env.summary["node_mapping"]["facilities"]

        self._entity_dict: Dict[int, SupplyChainEntity] = {
            entity.id: entity
            for entity in self._env.business_engine.get_entity_list()
        }

        product_infos, pid2idx, cid2pid, cid2fid, sidx2pidx = self._extract_facility_and_product_info()

        self.product_infos: List[GlobalProductInfo] = product_infos
        self.product_id2idx_in_product_infos: Dict[int, int] = pid2idx

        # consumer unit id to product unit id.
        self.consumer_id2product_id: Dict[int, int] = cid2pid
        # consumer unit id to facility id.
        self.consumer_id2facility_id: Dict[int, int] = cid2fid

        # seller node/data_model index to product node/data_model index.
        self.seller_index2product_index: Dict[int, int] = sidx2pidx

        self.num_products = len(self.product_infos)
        self.num_facilities = len(self._facility_info_dict)

        # NOTE: ordered list only valid for DAG-Topology.
        # self._ordered_products: List[GlobalProductInfo] = self._get_products_sorted_from_downstreams_to_upstreams()

        self.accumulated_balance_sheet = defaultdict(int)

    def update_env(self, env: Env) -> None:
        self._env = env

    def _extract_facility_and_product_info(self) -> Tuple[
        List[GlobalProductInfo], Dict[int, int], Dict[int, int], Dict[int, int],
    ]:
        product_infos: List[GlobalProductInfo] = []
        product_id2idx_in_product_infos: Dict[int, int] = {}

        consumer_id2product_id: Dict[int, int] = {}
        consumer_id2facility_id: Dict[int, int] = {}

        seller_index2product_index: Dict[int, int] = {}

        for facility_id, facility_info in self._facility_info_dict.items():
            distribution_info: DistributionUnitInfo = facility_info.distribution_info
            storage_info: StorageUnitInfo = facility_info.storage_info
            downstreams: Dict[int, List[int]] = facility_info.downstreams
            upstreams: Dict[int, List[int]] = facility_info.upstreams

            for sku_id, product_info in facility_info.products_info.items():
                product_id2idx_in_product_infos[product_info.id] = len(product_infos)

                if product_info.consumer_info:
                    consumer_id2product_id[product_info.consumer_info.id] = product_info.id
                    consumer_id2facility_id[product_info.consumer_info.id] = facility_id

                if product_info.seller_info:
                    seller_index2product_index[product_info.seller_info.node_index] = product_info.node_index

                product_infos.append(
                    GlobalProductInfo(
                        unit_id=product_info.id,
                        sku_id=product_info.sku_id,
                        node_index=product_info.node_index,
                        storage_index=storage_info.node_index if storage_info else None,
                        distribution_index=distribution_info.node_index if distribution_info else None,
                        consumer_index=product_info.consumer_info.node_index if product_info.consumer_info else None,
                        seller_index=product_info.seller_info.node_index if product_info.seller_info else None,
                        manufacture_index=(
                            product_info.manufacture_info.node_index if product_info.manufacture_info else None
                        ),
                        downstream_product_id_list=[
                            self._facility_info_dict[fid].products_info[sku_id].id for fid in downstreams[sku_id]
                        ] if sku_id in downstreams else [],
                        upstream_product_id_list=[
                            self._facility_info_dict[fid].products_info[sku_id].id for fid in upstreams[sku_id]
                        ] if sku_id in upstreams else [],
                    )
                )

        return (
            product_infos, product_id2idx_in_product_infos,
            consumer_id2product_id, consumer_id2facility_id, seller_index2product_index
        )

    def _get_products_sorted_from_downstreams_to_upstreams(self) -> List[GlobalProductInfo]:
        # Topological sorting
        ordered_products: List[GlobalProductInfo] = []
        product_info_dict = {product.unit_id: product for product in self.product_infos}

        down_degree = Counter()
        for product_info in self.product_infos:
            down_degree[product_info.unit_id] += len(product_info.downstream_product_id_list)

        queue = deque()
        for unit_id in product_info_dict.keys():
            if down_degree[unit_id] == 0:
                queue.append(unit_id)

        while queue:
            unit_id = queue.popleft()
            product_info = product_info_dict[unit_id]
            ordered_products.append(product_info)
            for upstream_unit_id in product_info.upstream_product_id_list:
                down_degree[upstream_unit_id] -= 1
                if down_degree[upstream_unit_id] == 0:
                    queue.append(upstream_unit_id)

        return ordered_products

    def _calc_consumer(self, tick: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate ConsumerUnit's step_cost and get comsumer id list.

        Returns:
            np.ndarray: consumer id list, with consumer node index as 1st-dim index.
            np.ndarray: consumer step cost, with consumer node index as 1st-dim index.
        """
        consumer_ids = get_attributes(self._env, "consumer", "id", tick).astype(np.int)

        # order_base_cost + order_product_cost
        consumer_step_cost = -1 * (
            get_attributes(self._env, "consumer", "order_base_cost", tick)
            + get_attributes(self._env, "consumer", "order_product_cost", tick)
        )

        return consumer_ids, consumer_step_cost

    def _calc_seller(self, tick: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate SellerUnit's step_profit and step_cost.

        Returns:
            np.ndarray: seller step profit, with seller node index as 1st-dim index.
            np.ndarray: seller step cost, with seller node index as 1st-dim index.
        """
        price = self._env.snapshot_list["product"][
            self._env.business_engine.frame_index(tick):[
                self.seller_index2product_index[sidx] for sidx in range(len(self._env.snapshot_list["seller"]))
            ]:"price"
        ].flatten()

        # profit = sold * price
        seller_step_profit = get_attributes(self._env, "seller", "sold", tick) * price

        # loss = demand * price * backlog_ratio
        seller_step_cost = -1 * (
            (get_attributes(self._env, "seller", "demand", tick) - get_attributes(self._env, "seller", "sold", tick))
            * get_attributes(self._env, "seller", "backlog_ratio", tick)
            * price
        )

        return seller_step_profit, seller_step_cost

    def _calc_manufacture(self, tick: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate ManufactureUnit's step_cost and get manufacture id list.

        Returns:
            np.ndarray: manufacture id list, with manufacture node index as 1st-dim index.
            np.ndarray: manufacture step cost, with manufacture node index as 1st-dim index.
        """
        manufacture_ids = get_attributes(self._env, "manufacture", "id", tick).astype(np.int)

        # loss = manufacture number * cost
        manufacture_step_cost = -1 * get_attributes(self._env, "manufacture", "manufacture_cost", tick)

        return manufacture_ids, manufacture_step_cost

    def _calc_product_storage(self, tick: int) -> List[Dict[int, float]]:
        """Calculate product storage cost based on infos from StorageUnit.

        Returns:
            List[Dict[int, float]]: product storage step cost dict, with storage node index as list index and sku id as
                the keys in dict.
        """
        sku_id_lists = get_list_attributes(self._env, "storage", "sku_id_list", tick)
        product_id_lists = get_list_attributes(self._env, "storage", "product_id_list", tick)
        quantity_lists = get_list_attributes(self._env, "storage", "product_quantity", tick)

        # Idx: storage node index. Key: sku_id, value: cost.
        product_storage_step_cost: List[Dict[int, float]] = [
            {
                sku_id: -quantity * self._entity_dict[product_id].skus.unit_storage_cost
                for sku_id, product_id, quantity in zip(
                    sku_id_list.astype(np.int), product_id_list.astype(np.int), quantity_list.astype(np.int)
                )
            }
            for sku_id_list, product_id_list, quantity_list in zip(sku_id_lists, product_id_lists, quantity_lists)
        ]

        return product_storage_step_cost

    def _calc_product_distribution(self, tick: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate ProductUnit's step distribution profit and step distribution cost.

        Returns:
            np.ndarray: product step distribution profit, with product node index as 1st-dim index.
            np.ndarray: product step distribution cost, with product node index as 1st-dim index.
        """
        # product distribution profit = check order * price
        product_step_distribution_profit = (
            get_attributes(self._env, "product", "check_in_quantity_in_order", tick)
            * get_attributes(self._env, "product", "price", tick)
        )

        # product distribution loss = transportation cost + delay order penalty
        product_step_distribution_cost = -1 * (
            get_attributes(self._env, "product", "transportation_cost", tick)
            + get_attributes(self._env, "product", "delay_order_penalty", tick)
        )

        return product_step_distribution_profit, product_step_distribution_cost

    def _calc_product(
        self,
        consumer_step_cost: np.ndarray,
        manufacture_step_cost: np.ndarray,
        seller_step_profit: np.ndarray,
        seller_step_cost: np.ndarray,
        storage_product_step_cost: List[Dict[int, float]],
        product_distribution_step_profit: np.ndarray,
        product_distribution_step_cost: np.ndarray,
        tick: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ProductUnit's step profit, step cost and step balance.

        Returns:
            np.ndarray: product step profit, with product node index as 1st-dim index.
            np.ndarray: product step cost, with product node index as 1st-dim index.
            np.ndarray: product step balance, with product node index as 1st-dim index.
        """
        product_step_profit = np.zeros(self.num_products)
        product_step_cost = np.zeros(self.num_products)

        # product = consumer + seller + manufacture + storage + distribution (+ downstreams)
        # NOTE: it must be self._ordered_products if we want to take its downstreams into consideration.
        # Otherwise, the order does not matter.
        # NOTE: It is not a DAG-topology in SCI, so that we cannot use ordered one here.
        # for product in self._ordered_products:
        for product in self.product_infos:
            i = product.node_index

            if product.consumer_index is not None:
                product_step_cost[i] += consumer_step_cost[product.consumer_index]

            if product.seller_index is not None:
                product_step_profit[i] += seller_step_profit[product.seller_index]
                product_step_cost[i] += seller_step_cost[product.seller_index]

            if product.manufacture_index is not None:
                product_step_cost[i] += manufacture_step_cost[product.manufacture_index]

            if product.storage_index is not None:
                product_step_cost[i] += storage_product_step_cost[product.storage_index][product.sku_id]

            if product.distribution_index is not None:
                product_step_profit[i] += product_distribution_step_profit[i]
                product_step_cost[i] += product_distribution_step_cost[i]

            # TODO: Check if we still need to consider the downstream profit & cost.
            # for down_product_id in product.downstream_product_id_list:
            #     product_step_profit[i] += product_step_profit[
            #         self.product_infos[self.product_id2idx_in_product_infos[down_product_id]].node_index
            #     ]
            #     product_step_cost[i] += product_step_cost[
            #         self.product_infos[self.product_id2idx_in_product_infos[down_product_id]].node_index
            #     ]

        product_step_balance = product_step_profit + product_step_cost

        return product_step_profit, product_step_cost, product_step_balance

    def _calc_facility(
        self,
        product_step_profit: np.ndarray,
        product_step_cost: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Facility's step profit, step cost and step balance.

        Args:
            product_step_profit (np.ndarray): product step profit, with product node index as 1st-dim index.
            product_step_cost (np.ndarray): product step cost, with product node index as 1st-dim index.

        Returns:
            np.ndarray: facility step profit, with the default index in self._facility_info_dict as 1st-dim index.
            np.ndarray: facility step cost, with the default index in self._facility_info_dict as 1st-dim index.
            np.ndarray: facility step balance, with the default index in self._facility_info_dict as 1st-dim index.
        """
        facility_step_profit = np.zeros(self.num_facilities)
        facility_step_cost = np.zeros(self.num_facilities)

        for i, facility_info in enumerate(self._facility_info_dict.values()):
            for product_info in facility_info.products_info.values():
                facility_step_profit[i] += product_step_profit[product_info.node_index]
                facility_step_cost[i] += product_step_cost[product_info.node_index]

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
        """Update balance sheet according to pre-defined reward definition.

        Args:
            product_step_balance (np.ndarray): product step balance, with product node index as 1st-dim index.
            consumer_ids (np.ndarray): consumer id list, with consumer node index as 1st-dim index.
            manufacture_ids (np.ndarray): manufacture id list, with manufacture node index as 1st-dim index.
            manufacture_step_cost (np.ndarray): manufacture step cost, with manufacture node index as 1st-dim index.
            facility_step_balance (np.ndarray): facility step balance, with the default index in
                self._facility_info_dict as 1st-dim index.

        Returns:
            Dict[int, Tuple[float, float]]: (step balance, step reward) for each entity, with entity id as the key.
                Currently, the entity includes: ProductUnit, ConsumerUnit, ManufactureUnit and Facility.
        """

        # Key: the facility/unit id; Value: (balance, reward).
        balance_and_reward: Dict[int, Tuple[float, float]] = {}

        # For Product Units.
        for product_info in self.product_infos:
            id_ = product_info.unit_id
            balance = product_step_balance[product_info.node_index]

            balance_and_reward[id_] = (balance, balance)
            self.accumulated_balance_sheet[id_] += balance

        # For Consumer Units. Let it be equal to the ProductUnit's it belongs to.
        for id_ in consumer_ids:
            balance_and_reward[id_] = balance_and_reward[self.consumer_id2product_id[id_]]
            self.accumulated_balance_sheet[id_] += balance_and_reward[id_][0]

        # For Manufacture Units. balance = cost (+ 0-profit).
        for id_, balance in zip(manufacture_ids, manufacture_step_cost):
            balance_and_reward[id_] = (balance, balance)
            self.accumulated_balance_sheet[id_] += balance

        for id_, balance in zip(self._facility_info_dict.keys(), facility_step_balance):
            balance_and_reward[id_] = (balance, balance)
            self.accumulated_balance_sheet[id_] += balance

        # team reward
        if self._team_reward:
            team_reward = defaultdict(lambda: (0, 0))
            for id_ in consumer_ids:
                facility_id = self.consumer_id2facility_id[id_]
                team_reward[facility_id] = (
                    team_reward[facility_id][0] + balance_and_reward[id_][0],
                    team_reward[facility_id][1] + balance_and_reward[id_][1]
                )

            for id_ in consumer_ids:
                facility_id = self.consumer_id2facility_id[id_]
                balance_and_reward[id_] = team_reward[facility_id]

        return balance_and_reward

    def calc_and_update_balance_sheet(self, tick: int) -> Dict[int, Tuple[float, float]]:
        """Calculate and update balance sheet.

        Returns:
            Dict[int, Tuple[float, float]]: (step balance, step reward) for each entity, with entity id as the key.
                Currently, the entity includes: ProductUnit, ConsumerUnit, ManufactureUnit and Facility.
        """
        # TODO: Add cache for each tick.
        # TODO: Add logic to confirm the balance sheet for the same tick would not be re-calculate.

        # Basic Units: profit & cost
        consumer_ids, consumer_step_cost = self._calc_consumer(tick)
        seller_step_profit, seller_step_cost = self._calc_seller(tick)
        manufacture_ids, manufacture_step_cost = self._calc_manufacture(tick)

        product_storage_step_cost = self._calc_product_storage(tick)
        product_distribution_step_profit, product_distribution_step_cost = self._calc_product_distribution(tick)

        # Product: profit, cost & balance
        product_step_profit, product_step_cost, product_step_balance = self._calc_product(
            consumer_step_cost,
            manufacture_step_cost,
            seller_step_profit,
            seller_step_cost,
            product_storage_step_cost,
            product_distribution_step_profit,
            product_distribution_step_cost,
            tick
        )

        _, _, facility_step_balance = self._calc_facility(product_step_profit, product_step_cost)

        balance_and_reward = self._update_balance_sheet(
            product_step_balance, consumer_ids, manufacture_ids, manufacture_step_cost, facility_step_balance
        )

        return balance_and_reward
