# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple

import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain.facilities.facility import FacilityInfo
from maro.simulator.scenarios.supply_chain.units.distribution import DistributionUnitInfo
from maro.simulator.scenarios.supply_chain.units.storage import StorageUnitInfo


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
        "vehicle_index_list",
    ),
)


class BalanceSheetCalculator:
    consumer_features = ("id", "purchased", "received",
                         "order_cost", "order_product_cost")
    seller_features = ("id", "sold", "demand", "price", "backlog_ratio")
    manufacture_features = ("id", "manufacture_quantity", "unit_product_cost")
    product_features = (
        "id", "price", 'check_in_quantity_in_order', 'delay_order_penalty', "transportation_cost")
    storage_features = ("capacity", "remaining_space")

    def __init__(self, env: Env, team_reward) -> None:
        self._env: Env = env
        self._team_reward = team_reward

        facilities, products, pid2idx, cid2pid = self._extract_facility_and_product_info()

        self.facility_levels: List[FacilityLevelInfo] = facilities
        self.products: List[ProductInfo] = products

        self.product_id2idx: Dict[int, int] = pid2idx
        self.consumer_id2product_id: Dict[int, int] = cid2pid

        self.num_products = len(self.products)
        self.num_facilities = len(self.facility_levels)

        self._ordered_products: List[ProductInfo] = self._get_products_sorted_from_downstreams_to_upstreams()

        self.accumulated_balance_sheet = defaultdict(int)
        self.product_metric_track = {}
        self.tick_cached = set()
        self.facility_info = self._env.summary['node_mapping']['facilities']
        self.sku_meta_info = self._env.summary['node_mapping']['skus']
        for col in ['id', 'sku_id', 'facility_id', 'facility_name', 'name', 'tick', 'inventory_in_stock', 'unit_inventory_holding_cost']:
            self.product_metric_track[col] = []
        for (name, cols) in [("consumer", self.consumer_features),
                             ("seller", self.seller_features),
                             ("manufacture", self.manufacture_features),
                             ("product", self.product_features)]:
            for fea in cols:
                if fea == 'id':
                    continue
                self.product_metric_track[f"{name}_{fea}"] = []

    def _extract_facility_and_product_info(self) -> Tuple[
        List[FacilityLevelInfo], List[ProductInfo], Dict[int, int], Dict[int, int]
    ]:
        facility_levels: List[FacilityLevelInfo] = []
        products: List[ProductInfo] = []

        product_id2idx: Dict[int, int] = {}
        consumer_id2product_id: Dict[int, int] = {}

        facility_info_dict: Dict[int, FacilityInfo] = self._env.summary["node_mapping"]["facilities"]
        for facility_id, facility_info in facility_info_dict.items():
            distribution_info: DistributionUnitInfo = facility_info.distribution_info
            storage_info: StorageUnitInfo = facility_info.storage_info
            downstreams: Dict[int, List[int]] = facility_info.downstreams

            product_id_list = []
            for product_id, product_info in facility_info.products_info.items():
                product_id_list.append(product_info.id)
                product_id2idx[product_info.id] = len(products)

                if product_info.consumer_info:
                    consumer_id2product_id[product_info.consumer_info.id] = product_info.id

                products.append(
                    ProductInfo(
                        unit_id=product_info.id,
                        sku_id=product_info.product_id,
                        node_index=product_info.node_index,
                        facility_id = facility_id,
                        storage_index=storage_info.node_index,
                        distribution_index=distribution_info.node_index if distribution_info else None,
                        downstream_product_unit_id_list=[
                            facility_info_dict[fid].products_info[product_id].id for fid in downstreams[product_id]
                        ] if product_id in downstreams else [],
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
                    vehicle_index_list=distribution_info.vehicle_node_index_list if distribution_info else [],
                )
            )

        return facility_levels, products, product_id2idx, consumer_id2product_id

    def _get_products_sorted_from_downstreams_to_upstreams(self) -> List[ProductInfo]:
        # TODO: Sort products from downstream to upstream.
        ordered_products: List[ProductInfo] = []

        tmp_product_unit_dict = {product.unit_id: product for product in self.products}
        tmp_stack = []
        for product in self.products:
            # Skip if the product has already added.
            if tmp_product_unit_dict[product.unit_id] is None:
                continue

            # Add the downstreams to the stack.
            for product_unit_id in product.downstream_product_unit_id_list:
                tmp_stack.append(product_unit_id)

            # Insert current product to the head and mark it as already added.
            ordered_products.insert(0, product)
            tmp_product_unit_dict[product.unit_id] = None

            # Processing and add the downstreams of current product.
            while len(tmp_stack) > 0:
                downstream_product_unit_id = tmp_stack.pop()

                # Skip if it has already added.
                if tmp_product_unit_dict[downstream_product_unit_id] is None:
                    continue

                # Extract the downstream product unit.
                downstream_product_unit = tmp_product_unit_dict[downstream_product_unit_id]

                # Add the downstreams to the stack.
                for product_unit_id in downstream_product_unit.downstream_product_unit_id_list:
                    tmp_stack.append(product_unit_id)

                # Insert the unit to the head and mark it as already added.
                ordered_products.insert(0, downstream_product_unit)
                tmp_product_unit_dict[downstream_product_unit_id] = None

        return ordered_products

    def _check_attribute_keys(self, target_type: str, attribute: str) -> None:
        valid_target_types = list(self._env.summary["node_detail"].keys())
        assert target_type in valid_target_types, f"Target_type {target_type} not in {valid_target_types}!"

        valid_attributes = list(self._env.summary["node_detail"][target_type]["attributes"].keys())
        assert attribute in valid_attributes, (
            f"Attribute {attribute} not valid for {target_type}. Valid attributes: {valid_attributes}"
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

    def _calc_consumer(self, tick: int) -> tuple:
        consumer_ids = self._get_attributes("consumer", "id", tick).astype(np.int)

        # order_cost + order_product_cost
        consumer_step_cost = -1 * (
            self._get_attributes("consumer", "order_cost", tick)
            + self._get_attributes("consumer", "order_product_cost", tick)
        )

        return consumer_ids, consumer_step_cost

    def _calc_seller(self, tick: int) -> tuple:
        # profit = sold * price
        seller_step_profit = (
            self._get_attributes("seller", "sold", tick)
            * self._get_attributes("seller", "price", tick)
        )

        # loss = demand * price * backlog_ratio
        seller_step_cost = -1 * (
            (self._get_attributes("seller", "demand", tick) - self._get_attributes("seller", "sold", tick))
            * self._get_attributes("seller", "price", tick)
            * self._get_attributes("seller", "backlog_ratio", tick)
        )

        return seller_step_profit, seller_step_cost

    def _calc_manufacture(self, tick: int) -> tuple:
        manufacture_ids = self._get_attributes("manufacture", "id", tick).astype(np.int)

        # loss = manufacture number * cost
        manufacture_step_cost = -1 * (
            self._get_attributes("manufacture", "manufacture_quantity", tick)
            * self._get_attributes("manufacture", "unit_product_cost", tick)
        )

        return manufacture_ids, manufacture_step_cost

    def _calc_storage(self, tick: int) -> tuple:
        unit_storage_cost = self._get_list_attributes("storage", "unit_storage_cost", tick)

        # loss = (capacity - remaining space) * cost
        facility_storage_step_cost = [
            -((_capacity - _remaining) * _cost).sum()
            for _capacity, _remaining, _cost in zip(
                self._get_list_attributes("storage", "capacity", tick),
                self._get_list_attributes("storage", "remaining_space", tick),
                unit_storage_cost,
            )
        ]

        storage_product_step_cost: List[Dict[int, float]] = [
            {
                product_id: -quantity * unit_cost_list[storage_index]
                for product_id, quantity, storage_index in zip(id_list, quantity_list, storage_index_list)
            }
            for id_list, quantity_list, storage_index_list, unit_cost_list in zip(
                [il.astype(np.int) for il in self._get_list_attributes("storage", "product_list", tick)],
                [ql.astype(np.int) for ql in self._get_list_attributes("storage", "product_quantity", tick)],
                [si.astype(np.int) for si in self._get_list_attributes("storage", "product_storage_index", tick)],
                [uc.astype(np.float) for uc in unit_storage_cost],
            )
        ]

        return facility_storage_step_cost, storage_product_step_cost

    def _calc_vehicle(self, tick: int) -> tuple:
        # loss = cost * payload
        vehicle_step_cost = -1 * (
            self._get_attributes("vehicle", "payload", tick)
            * self._get_attributes("vehicle", "unit_transport_cost", tick)
        )
        return vehicle_step_cost

    def _calc_product_distribution(self, tick: int) -> tuple:
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
        self, consumer_step_cost, manufacture_step_cost, seller_step_profit, seller_step_cost,
        storage_product_step_cost, product_distribution_step_profit, product_distribution_step_cost, tick
    ) -> tuple:
        product_step_profit = np.zeros(self.num_products)
        product_step_cost = np.zeros(self.num_products)

        # product = consumer + seller + manufacture + storage + distribution + downstreams
        for product in self._ordered_products:
            i = product.node_index
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
                        val = self._get_attributes("product", fea, tick)[i]
                        self.product_metric_track[f"product_{fea}"].append(val)
                for fea in self.consumer_features:
                    if fea != 'id':
                        val = (self._get_attributes("consumer", fea, tick)[product.consumer_index] if product.consumer_index else 0)
                        self.product_metric_track[f"consumer_{fea}"].append(val)
                for fea in self.seller_features:
                    if fea != 'id':
                        val = (self._get_attributes("seller", fea, tick)[product.seller_index] if product.seller_index else 0)
                        self.product_metric_track[f"seller_{fea}"].append(val)
                for fea in self.manufacture_features:
                    if fea != 'id':
                        val = (self._get_attributes("manufacture", fea, tick)[product.manufacture_index] if product.manufacture_index else 0)
                        self.product_metric_track[f"manufacture_{fea}"].append(val)
                self.product_metric_track['unit_inventory_holding_cost'].append(self._get_list_attributes("storage", "unit_storage_cost", tick)[product.storage_index])
                product_idx_in_storage = np.where(self._get_list_attributes("storage", "product_list", tick)[product.storage_index] == product.sku_id)[0]
                self.product_metric_track['inventory_in_stock'].append(self._get_list_attributes("storage", "product_quantity", tick)[product.storage_index][product_idx_in_storage])

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
        self, product_step_profit, product_step_cost, tick
    ) -> tuple:
        facility_step_profit = np.zeros(self.num_facilities)
        facility_step_cost = np.zeros(self.num_facilities)

        for i, facility in enumerate(self.facility_levels):
            # TODO: check is it still needed, since we already add it into the product
            # facility_step_cost[i] += storage_step_cost[facility.storage_index]

            # TODO: check is it still needed, since we already add it into the product.
            # Also, pending penalty not included here.
            # for vidx in facility.vehicle_index_list:
            #     facility_step_cost[i] += vehicle_step_cost[vidx]

            for pid in facility.product_unit_id_list:
                facility_step_profit[i] += product_step_profit[self.product_id2idx[pid]]
                facility_step_cost[i] += product_step_cost[self.product_id2idx[pid]]

        facility_step_balance = facility_step_profit + facility_step_cost

        return facility_step_profit, facility_step_cost, facility_step_balance

    def _update_balance_sheet(
        self, product_step_balance, consumer_ids, manufacture_ids, manufacture_step_cost, facility_step_balance
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
        facility_storage_step_cost, storage_product_step_cost = self._calc_storage(tick)
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
            tick
        )

        _, _, facility_step_balance = self._calc_facility(product_step_profit, product_step_cost, tick
        )

        balance_and_reward = self._update_balance_sheet(
            product_step_balance, consumer_ids, manufacture_ids, manufacture_step_cost, facility_step_balance
        )

        return balance_and_reward

    def reset(self):
        for key in self.product_metric_track.keys():
            self.product_metric_track[key] = []
        self.tick_cached.clear()
            