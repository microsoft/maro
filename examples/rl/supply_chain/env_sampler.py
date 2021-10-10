# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from collections import defaultdict, namedtuple
from os.path import dirname
from typing import List

import scipy.stats as st
import numpy as np

from maro.rl.learning import AbsEnvSampler
from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ConsumerAction, ManufactureAction 

sc_path = dirname(__file__)
sys.path.insert(0, sc_path)
from config import NUM_RL_POLICIES, distribution_features, env_conf, keys_in_state, seller_features, sku_agent_types
from env_info import get_env_info
from policies import or_policy_func_dict, policy_func_dict


class SCEnvSampler(AbsEnvSampler):
    def __init__(self, get_env, get_policy_func_dict, agent2policy):
        super().__init__(get_env, get_policy_func_dict, agent2policy)
        self.env = self._learn_env
        self.balance_cal = BalanceSheetCalculator(self.env)
        self.cur_balance_sheet_reward = None

        self._summary = self.env.summary['node_mapping']
        self._configs = self.env.configs
        self._agent_types = self._summary["agent_types"]
        self._units_mapping = self._summary["unit_mapping"]
        self._agent_list = self.env.agent_idx_list

        self._sku_number = len(self._summary["skus"]) + 1
        self._max_sources_per_facility = self._summary["max_sources_per_facility"]

        # state for each tick
        self._cur_metrics = self.env.metrics
        # cache for ppf value.
        self._service_index_ppf_cache = {}

        # facility -> {
        #   data_model_index:int,
        #   storage:UnitBaseInfo,
        #   distribution: UnitBaseInfo,
        #   sku_id: {
        #       skuproduct: UnitBaseInfo,
        #       consumer: UnitBaseInfo,
        #       seller: UnitBaseInfo,
        #       manufacture: UnitBaseInfo
        #   }
        # }
        self._env_info = get_env_info(self.env)
        # facility id -> in_transit_orders
        self._facility_in_transit_orders = {}
        # current distribution states
        self._cur_distribution_states = None
        # current consumer states
        self._cur_consumer_states = None
        # current seller states
        self._cur_seller_states = None

        self.stock_status = {}
        self.demand_status = {}
        # key: product unit id, value: number
        self.orders_from_downstreams = {}
        self.consumer_orders = {}
        self.order_in_transit_status = {}
        self.order_to_distribute_status = {}

    def get_or_policy_state(self, state, agent_info):
        state = {'is_facility': not (agent_info.agent_type in sku_agent_types)}
        if agent_info.is_facility:
            return state

        if agent_info.agent_type in ["product", "productstore"]:
            product_unit_id = agent_info.id
        else:
            product_unit_id = agent_info.parent_id
        unit_storage_cost = self.balance_cal.products[self.balance_cal.product_id2index_dict[product_unit_id]][4]

        product_metrics = self._cur_metrics["products"][product_unit_id]
        state['sale_mean'] = product_metrics["sale_mean"]
        state['sale_std'] = product_metrics["sale_std"]

        facility = self._env_info["facility_levels"][agent_info.facility_id]
        state['unit_storage_cost'] = unit_storage_cost
        state['order_cost'] = 1
        product_info = facility[agent_info.sku.id]
        if "consumer" in product_info:
            idx = product_info["consumer"].node_index
            state['order_cost'] = self.env.snapshot_list["consumer"][self.env.tick:idx:"order_cost"].flatten()[0]
        state['storage_capacity'] = facility['storage'].config["capacity"]
        state['storage_levels'] = self._env_info["storage_product_num"][agent_info.facility_id]
        state['consumer_in_transit_orders'] = self._facility_in_transit_orders[agent_info.facility_id]
        state['product_idx'] = self._env_info["storage_product_indexes"][agent_info.facility_id][agent_info.sku.id] + 1
        state['vlt'] = agent_info.sku.vlt
        state['service_level'] = agent_info.sku.service_level
        return state

    def get_rl_policy_state(self, state, agent_info):
        self._update_facility_features(state, agent_info)
        self._update_storage_features(state, agent_info)
        # bom do not need to update
        # self._add_bom_features(state, agent_info)
        self._update_distribution_features(state, agent_info)
        self._update_sale_features(state, agent_info)
        # vlt do not need to update
        # self._update_vlt_features(state, agent_info)
        self._update_consumer_features(state, agent_info)
        # self._add_price_features(state, agent_info)
        self._update_global_features(state)

        self.stock_status[agent_info.id] = state['inventory_in_stock']

        self.demand_status[agent_info.id] = state['sale_hist'][-1]

        self.order_in_transit_status[agent_info.id] = state['inventory_in_transit']

        self.order_to_distribute_status[agent_info.id] = state['distributor_in_transit_orders_qty']

        np_state = self._serialize_state(state)
        return np_state

    def get_state(self, tick=None):
        if tick is None:
            tick = self.env.tick
        settings: dict = self.env.configs.settings
        consumption_hist_len = settings['consumption_hist_len']
        hist_len = settings['sale_hist_len']
        consumption_ticks = [tick - i for i in range(consumption_hist_len-1, -1, -1)]
        hist_ticks = [tick - i for i in range(hist_len-1, -1, -1)]

        self.cur_balance_sheet_reward = self.balance_cal.calc()
        self._cur_metrics = self.env.metrics

        self._cur_distribution_states = self.env.snapshot_list["distribution"][tick::distribution_features] \
            .flatten() \
            .reshape(-1, len(distribution_features)) \
            .astype(np.int)

        self._cur_consumer_states = self.env.snapshot_list["consumer"][consumption_ticks::"latest_consumptions"] \
            .flatten() \
            .reshape(-1, len(self.env.snapshot_list["consumer"]))

        self._cur_seller_states = self.env.snapshot_list["seller"][hist_ticks::seller_features].astype(np.int)

        # facility level states
        for facility_id in self._env_info["facility_product_utilization"]:
            # reset for each step
            self._env_info["facility_product_utilization"][facility_id] = 0

            in_transit_orders = self._cur_metrics['facilities'][facility_id]["in_transit_orders"]

            self._facility_in_transit_orders[facility_id] = [0] * self._sku_number

            for sku_id, number in in_transit_orders.items():
                self._facility_in_transit_orders[facility_id][sku_id] = number

        final_state = {}

        # calculate storage info first, then use it later to speed up.
        for facility_id, storage_index in self._env_info["facility2storage"].items():
            product_numbers = self.env.snapshot_list["storage"][tick:storage_index:"product_number"] \
                .flatten() \
                .astype(np.int)

            for pid, index in self._env_info["storage_product_indexes"][facility_id].items():
                product_number = product_numbers[index]

                self._env_info["storage_product_num"][facility_id][pid] = product_number
                self._env_info["facility_product_utilization"][facility_id] += product_number

        for agent_info in self._agent_list:
            state = self._env_info["placeholder_state"][agent_info.id]

            # storage_index = self._env_info["facility2storage"][agent_info.facility_id]

            np_state = self.get_rl_policy_state(state, agent_info)
            if agent_info.agent_type == "producer":
                np_state = self.get_or_policy_state(state, agent_info)

            # agent_info.agent_type -> policy
            final_state[f"{agent_info.agent_type}.{agent_info.id}"] = np_state

        #self.reward_status = {f_id: np.float32(reward[1]) for f_id, reward in self.cur_balance_sheet_reward.items()}
        #self.balance_status = {f_id: np.float32(reward[0]) for f_id, reward in self.cur_balance_sheet_reward.items()}

        return final_state

    def get_reward(self, actions, tick):
        # get related product, seller, consumer, manufacture unit id
        # NOTE: this mapping does not contain facility id, so if id is not exist, then means it is a facility
        # product_unit_id, facility_id, seller_id, consumer_id, producer_id = self._env_info["unit2product"][id]
        # return {
        #     f"{self._env_info["agentid2info"][f_id].agent_type}.{f_id}": np.float32(bwt[1]) / np.float32(self._configs.settings["reward_normalization"])
        #     for f_id, bwt in self.cur_balance_sheet_reward.items()
        # }
        self.cur_balance_sheet_reward = self.balance_cal.calc()
        rewards = defaultdict(float)
        for f_id, bwt in self.cur_balance_sheet_reward.items():
            agent = self._env_info["agentid2info"][f_id]
            if agent.agent_type == 'consumer':
                rewards[f"{self._env_info['agentid2info'][f_id].agent_type}.{f_id}"] = np.float32(bwt[1]) / np.float32(self._configs.settings["reward_normalization"])
            else:
                rewards[f"{self._env_info['agentid2info'][f_id].agent_type}.{f_id}"] = 0

        return rewards

    def get_env_actions(self, action_by_agent):
        # cache the sources for each consumer if not yet cached
        if not hasattr(self, "consumer2source"):
            self.consumer2source, self.consumer2product = {}, {}
            for facility in self.env.summary["node_mapping"]["facilities"].values():
                products = facility["units"]["products"]
                for product_id, product in products.items():
                    consumer = product["consumer"]
                    if consumer is not None:
                        consumer_id = consumer["id"]
                        self.consumer2source[consumer_id] = consumer["sources"]
                        self.consumer2product[consumer_id] = product_id

        env_action = []
        for agent_id, action in action_by_agent.items():
            unit_id = int(agent_id.split(".")[1])

            is_facility = unit_id not in self._units_mapping

            # ignore facility to reduce action number
            if is_facility:
                continue

            # consumer action
            if agent_id.startswith("consumer"):
                product_id = self.consumer2product.get(unit_id, 0)
                sources = self.consumer2source.get(unit_id, [])
                if sources:
                    source_id = sources[0]
                    product_unit_id = self._env_info["unit2product"][unit_id][0]
                    action_number = int(int(action) * self._cur_metrics["products"][product_unit_id]["sale_mean"])

                    # ignore 0 quantity to reduce action number
                    if action_number == 0:
                        continue

                    sku = self._units_mapping[unit_id][3]

                    reward_discount = 1

                    env_action.append(ConsumerAction(
                        unit_id,
                        product_id,
                        source_id,
                        action_number,
                        sku.vlt,
                        reward_discount
                    ))

                    self.consumer_orders[product_unit_id] = action_number
                    self.orders_from_downstreams[self._env_info["facility_levels"][source_id][product_id]["skuproduct"].id] = action_number

            # manufacturer action
            elif agent_id.startswith("producer"):
                sku = self._units_mapping[unit_id][3]
                action = sku.production_rate

                # ignore invalid actions
                if action is None or action == 0:
                    continue

                env_action.append(ManufactureAction(unit_id, action))

        return env_action

    def _update_facility_features(self, state, agent_info):
        state['is_positive_balance'] = 1 if self.balance_cal.total_balance_sheet[agent_info.id] > 0 else 0

    def _update_storage_features(self, state, agent_info):
        facility_id = agent_info.facility_id
        state['storage_utilization'] = 0

        state['storage_levels'] = self._env_info["storage_product_num"][facility_id]
        state['storage_utilization'] = self._env_info["facility_product_utilization"][facility_id]

    def _update_sale_features(self, state, agent_info):
        if agent_info.agent_type not in sku_agent_types:
            return

        # Get product unit id for current agent.
        product_unit_id = agent_info.id if agent_info.agent_type in ["product", "productstore"] else agent_info.parent_id

        product_metrics = self._cur_metrics["products"][product_unit_id]

        state['sale_mean'] = product_metrics["sale_mean"]
        state['sale_std'] = product_metrics["sale_std"]

        facility = self._env_info["facility_levels"][agent_info.facility_id]
        product_info = facility[agent_info.sku.id]

        if "seller" not in product_info:
            # TODO: why gamma sale as mean?
            state['sale_gamma'] = state['sale_mean']

        if "consumer" in product_info:
            consumer_index = product_info["consumer"].node_index

            state['consumption_hist'] = list(
                self._cur_consumer_states[:, consumer_index])
            state['pending_order'] = list(
                product_metrics["pending_order_daily"])

        if "seller" in product_info:
            seller_index = product_info["seller"].node_index

            seller_states = self._cur_seller_states[:, seller_index, :]

            # For total demand, we need latest one.
            state['total_backlog_demand'] = seller_states[:, 0][-1][0]
            state['sale_hist'] = list(seller_states[:, 1].flatten())
            state['backlog_demand_hist'] = list(seller_states[:, 2])

    def _update_distribution_features(self, state, agent_info):
        facility = self._env_info["facility_levels"][agent_info.facility_id]
        distribution = facility.get("distribution", None)

        if distribution is not None:
            dist_states = self._cur_distribution_states[distribution.node_index]
            state['distributor_in_transit_orders'] = dist_states[1]
            state['distributor_in_transit_orders_qty'] = dist_states[0]

    def _update_consumer_features(self, state, agent_info):
        if agent_info.is_facility:
            return

        facility = self._env_info["facility_levels"][agent_info.facility_id]
        product_info = facility[agent_info.sku.id]

        # if "consumer" not in product_info:
        #     return

        state['consumer_in_transit_orders'] = self._facility_in_transit_orders[agent_info.facility_id]

        # FIX: we need plus 1 to this, as it is 0 based index, but we already aligned with 1 more
        # slot to use sku id as index ( 1 based).
        product_index = self._env_info["storage_product_indexes"][agent_info.facility_id][agent_info.sku.id] + 1
        state['inventory_in_stock'] = self._env_info["storage_product_num"][agent_info.facility_id][product_index]
        state['inventory_in_transit'] = state['consumer_in_transit_orders'][agent_info.sku.id]

        pending_order = self._cur_metrics["facilities"][agent_info.facility_id]["pending_order"]

        if pending_order is not None:
            state['inventory_in_distribution'] = pending_order[agent_info.sku.id]

        state['inventory_estimated'] = (state['inventory_in_stock']
                                        + state['inventory_in_transit']
                                        - state['inventory_in_distribution'])
        if state['inventory_estimated'] >= 0.5 * state['storage_capacity']:
            state['is_over_stock'] = 1

        if state['inventory_estimated'] <= 0:
            state['is_out_of_stock'] = 1

        service_index = state['service_level']

        if service_index not in self._service_index_ppf_cache:
            self._service_index_ppf_cache[service_index] = st.norm.ppf(
                service_index)

        ppf = self._service_index_ppf_cache[service_index]

        state['inventory_rop'] = (state['max_vlt'] * state['sale_mean']
                                  + np.sqrt(state['max_vlt']) * state['sale_std'] * ppf)

        if state['inventory_estimated'] < state['inventory_rop']:
            state['is_below_rop'] = 1

    def _update_global_features(self, state):
        state["global_time"] = self.env.tick

    def _serialize_state(self, state):
        result = []

        for norm, fields in keys_in_state:
            for field in fields:
                vals = state[field]
                if not isinstance(vals, list):
                    vals = [vals]
                if norm is not None:
                    vals = [max(0.0, min(20.0, x / (state[norm] + 0.01)))
                            for x in vals]
                result.extend(vals)

        return np.asarray(result, dtype=np.float32)


ProductInfo = namedtuple(
    "ProductInfo",
    (
        "unit_id",
        "sku_id",
        "node_index",
        "storage_index",
        "unit_storage_cost",
        "distribution_index",
        "downstream_product_units",
        "consumer_id_index_tuple",
        "seller_id_index_tuple",
        "manufacture_id_index_tuple"
    )
)

FacilityLevelInfo = namedtuple(
    "FacilityLevelInfo",
    (
        "unit_id",
        "product_unit_id_list",
        "storage_index",
        "unit_storage_cost",
        "distribution_index",
        "vehicle_index_list"
    )
)


class BalanceSheetCalculator:
    def __init__(self, env: Env):
        self.env = env
        self.products: List[ProductInfo] = []
        self.product_id2index_dict = {}
        self.facility_levels = []
        self.consumer_id2product = {}

        self.facilities = env.summary["node_mapping"]["facilities"]

        for facility_id, facility in self.facilities.items():
            pid_list = []
            distribution = facility["units"]["distribution"]

            for product_id, product in facility["units"]["products"].items():
                pid_list.append(product["id"])
                consumer = product["consumer"]
                if consumer is not None:
                    self.consumer_id2product[consumer["id"]] = product["id"]
                seller = product["seller"]
                manufacture = product["manufacture"]

                self.product_id2index_dict[product["id"]] = len(self.products)

                downstream_product_units = []
                downstreams = facility["downstreams"]

                if downstreams and len(downstreams) > 0 and product_id in downstreams:
                    for dfacility in downstreams[product_id]:
                        dproducts = self.facilities[dfacility]["units"]["products"]

                        downstream_product_units.append(dproducts[product_id]["id"])

                self.products.append(
                    ProductInfo(
                        unit_id=product["id"],
                        sku_id=product_id,
                        node_index=product["node_index"],
                        storage_index=facility["units"]["storage"]["node_index"],
                        unit_storage_cost=facility["units"]["storage"]["config"]["unit_storage_cost"],
                        distribution_index=distribution["node_index"] if distribution is not None else None,
                        downstream_product_units=downstream_product_units,
                        consumer_id_index_tuple=None if consumer is None else (consumer["id"], consumer["node_index"]),
                        seller_id_index_tuple=None if seller is None else (seller["id"], seller["node_index"]),
                        manufacture_id_index_tuple=None if manufacture is None else (manufacture["id"], manufacture["node_index"])
                    )
                )

            self.facility_levels.append(
                FacilityLevelInfo(
                    unit_id=facility_id,
                    product_unit_id_list=pid_list,
                    storage_index=facility["units"]["storage"]["node_index"],
                    unit_storage_cost=facility["units"]["storage"]["config"]["unit_storage_cost"],
                    distribution_index=distribution["node_index"] if distribution is not None else None,
                    vehicle_index_list=[
                        v["node_index"] for v in distribution["children"]
                    ] if distribution is not None else []
                )
            )

        # TODO: order products make sure calculate reward from downstream to upstream
        tmp_product_unit_dict = {}

        for product in self.products:
            tmp_product_unit_dict[product.unit_id] = product

        self._ordered_products = []

        tmp_stack = []

        for product in self.products:
            # skip if already being processed
            if tmp_product_unit_dict[product.unit_id] is None:
                continue

            for dproduct in product.downstream_product_units:
                # push downstream id to stack
                tmp_stack.append(dproduct)

            # insert current product to list head
            self._ordered_products.insert(0, product)
            # mark it as processed
            tmp_product_unit_dict[product.unit_id] = None

            while len(tmp_stack) > 0:
                # process downstream of product unit in stack
                dproduct_unit_id = tmp_stack.pop()

                # if it was processed then ignore
                if tmp_product_unit_dict[dproduct_unit_id] is None:
                    continue

                # or extract it downstreams
                dproduct_unit = tmp_product_unit_dict[dproduct_unit_id]

                dproduct_downstreams = dproduct_unit.downstream_product_units

                for dproduct in dproduct_downstreams:
                    tmp_stack.append(dproduct)

                # current unit in final list
                self._ordered_products.insert(0, dproduct_unit)
                tmp_product_unit_dict[dproduct_unit_id] = None

        self.total_balance_sheet = defaultdict(int)

        # tick -> (product unit id, sku id, manufacture number, manufacture cost, checkin order, delay penaty)
        self._supplier_reward_factors = {}

    def _check_attribute_keys(self, target_type: str, attribute: str):
        valid_target_types = list(self.env.summary["node_detail"].keys())
        assert target_type in valid_target_types, f"Target_type {target_type} not in {valid_target_types}!"

        valid_attributes = list(self.env.summary["node_detail"][target_type]["attributes"].keys())
        assert attribute in valid_attributes, (
            f"Attribute {attribute} not valid for {target_type}. "
            f"Valid attributes: {valid_attributes}"
        )
        return

    def _get_attributes(self, target_type: str, attribute: str, tick: int=None) -> np.ndarray:
        self._check_attribute_keys(target_type, attribute)

        if tick == None:
            tick = self.env.tick

        return self.env.snapshot_list[target_type][tick::attribute].flatten()

    def _get_list_attributes(self, target_type: str, attribute: str, tick: int=None) -> List[np.ndarray]:
        self._check_attribute_keys(target_type, attribute)

        if tick == None:
            tick = self.env.tick

        indexes = list(range(len(self.env.snapshot_list[target_type])))
        return [self.env.snapshot_list[target_type][tick:index:attribute].flatten() for index in indexes]

    def _calc_consumer(self):
        #### Consumer
        consumer_ids = self._get_attributes("consumer", "id").astype(np.int)

        # quantity * price
        order_profit = (
            self._get_attributes("consumer", "order_quantity")
            * self._get_attributes("consumer", "price")
        )

        # order_cost + order_product_cost
        consumer_step_balance_sheet_loss = -1 * (
            self._get_attributes("consumer", "order_cost")
            + self._get_attributes("consumer", "order_product_cost")
        )

        # consumer step reward: balance sheet los + profile * discount
        # consumer_step_reward = (
        #     consumer_step_balance_sheet_loss
        #     + order_profit * self._get_attributes("consumer", "reward_discount")
        # )
        consumer_step_reward = consumer_step_balance_sheet_loss

        consumer_step_balance_sheet = order_profit + consumer_step_balance_sheet_loss

        return consumer_ids, consumer_step_balance_sheet_loss, consumer_step_reward, consumer_step_balance_sheet

    def _calc_seller(self):
        #### Seller
        # profit = sold * price
        seller_balance_sheet_profit = (
            self._get_attributes("seller", "sold")
            * self._get_attributes("seller", "price")
        )

        # loss = demand * price * backlog_ratio
        seller_balance_sheet_loss = -1 * (
            self._get_attributes("seller", "demand")
            * self._get_attributes("seller", "price")
            * self._get_attributes("seller", "backlog_ratio")
        )

        # step reward = loss + profit
        seller_step_reward = seller_balance_sheet_loss + seller_balance_sheet_profit

        return seller_balance_sheet_profit, seller_balance_sheet_loss, seller_step_reward

    def _calc_manufacture(self):
        #### manufacture
        manufacture_ids = self._get_attributes("manufacture", "id").astype(np.int)

        # loss = manufacture number * cost
        manufacture_balance_sheet_loss = -1 * (
            self._get_attributes("manufacture", "manufacturing_number")
            * self._get_attributes("manufacture", "product_unit_cost")
        )

        # step reward = loss
        manufacture_step_reward = manufacture_balance_sheet_loss
        manufacture_step_balance_sheet = manufacture_balance_sheet_loss

        return manufacture_ids, manufacture_balance_sheet_loss, manufacture_step_reward, manufacture_step_balance_sheet

    def _calc_storage(self):
        #### storage
        # loss = (capacity-remaining space) * cost
        storage_balance_sheet_loss = -1 * (
            self._get_attributes("storage", "capacity")
            - self._get_attributes("storage", "remaining_space")
        )

        # create product number mapping for storages
        product_list = self._get_list_attributes("storage", "product_list")
        product_number = self._get_list_attributes("storage", "product_number")
        storages_product_map = {
            idx: {
                id: num
                for id, num in zip(id_list.astype(np.int), num_list.astype(np.int))
            }
            for idx, (id_list, num_list) in enumerate(zip(product_list, product_number))
        }

        return storage_balance_sheet_loss, storages_product_map

    def _calc_vehicle(self):
        ## vehicles
        # loss = cost * payload
        vehicle_balance_sheet_loss = -1 * (
            self._get_attributes("vehicle", "payload")
            * self._get_attributes("vehicle", "unit_transport_cost")
        )
        vehicle_step_reward = vehicle_balance_sheet_loss
        return vehicle_balance_sheet_loss, vehicle_step_reward

    def _calc_product_distribution(self):
        #### product
        # product distribution profit = check order * price
        product_distribution_balance_sheet_profit = (
            self._get_attributes("product", "distribution_check_order")
            * self._get_attributes("product", "price")
        )
        # product distribution loss = transportation cost + delay order penalty
        product_distribution_balance_sheet_loss = -1 * (
            self._get_attributes("product", "distribution_transport_cost")
            + self._get_attributes("product", "distribution_delay_order_penalty")
        )
        return product_distribution_balance_sheet_profit, product_distribution_balance_sheet_loss

    def _calc_product(
        self,
        consumer_step_balance_sheet_loss,
        consumer_step_reward,
        seller_balance_sheet_profit,
        seller_balance_sheet_loss,
        seller_step_reward,
        manufacture_balance_sheet_loss,
        manufacture_step_reward,
        storages_product_map,
        product_distribution_balance_sheet_profit,
        product_distribution_balance_sheet_loss,
    ):
        num_products = len(self.products)
        product_step_reward = np.zeros(num_products)
        product_balance_sheet_profit = np.zeros(num_products)
        product_balance_sheet_loss = np.zeros(num_products)

        # product = consumer + seller + manufacture + storage + distribution + downstreams
        for product in self._ordered_products:
            i = product.node_index

            if product.consumer_id_index_tuple:
                consumer_index = product.consumer_id_index_tuple[1]
                product_balance_sheet_loss[i] += consumer_step_balance_sheet_loss[consumer_index]
                product_step_reward[i] += consumer_step_reward[consumer_index]

            if product.seller_id_index_tuple:
                seller_index = product.seller_id_index_tuple[1]
                product_balance_sheet_profit[i] += seller_balance_sheet_profit[seller_index]
                product_balance_sheet_loss[i] += seller_balance_sheet_loss[seller_index]
                product_step_reward[i] += seller_step_reward[seller_index]

            if product.manufacture_id_index_tuple:
                manufacture_index = product.manufacture_id_index_tuple[1]
                product_balance_sheet_loss[i] += manufacture_balance_sheet_loss[manufacture_index]
                product_step_reward[i] += manufacture_step_reward[manufacture_index]

            storage_reward = -1 * storages_product_map[product.storage_index][product.sku_id] * product.unit_storage_cost
            product_step_reward[i] += storage_reward
            product_balance_sheet_loss[i] += storage_reward

            if product.distribution_index is not None:
                product_balance_sheet_profit[i] += product_distribution_balance_sheet_profit[i]
                product_balance_sheet_loss[i] += product_distribution_balance_sheet_loss[i]
                product_step_reward[i] += product_distribution_balance_sheet_loss[i] + product_distribution_balance_sheet_profit[i]

            if len(product.downstream_product_units) > 0:
                for did in product.downstream_product_units:
                    product_balance_sheet_profit[i] += product_balance_sheet_profit[self.product_id2index_dict[did]]
                    product_balance_sheet_loss[i] += product_balance_sheet_loss[self.product_id2index_dict[did]]
                    product_step_reward[i] += product_step_reward[self.product_id2index_dict[did]]

        product_balance_sheet = product_balance_sheet_profit + product_balance_sheet_loss

        return product_balance_sheet_profit, product_balance_sheet_loss, product_step_reward, product_balance_sheet

    def _calc_facility(
        self,
        storage_balance_sheet_loss,
        vehicle_balance_sheet_loss,
        product_balance_sheet_profit,
        product_balance_sheet_loss,
        product_step_reward
    ):
        num_facilities = len(self.facility_levels)
        facility_balance_sheet_loss = np.zeros(num_facilities)
        facility_balance_sheet_profit = np.zeros(num_facilities)
        facility_step_reward = np.zeros(num_facilities)

        # for facilities
        for i, facility in enumerate(self.facility_levels):
            # storage balance sheet
            # profit=0
            facility_balance_sheet_loss[i] += storage_balance_sheet_loss[facility.storage_index] * facility.unit_storage_cost

            # distribution balance sheet
            if facility.distribution_index is not None:
                for vidx in facility.vehicle_index_list:
                    facility_balance_sheet_loss[i] += vehicle_balance_sheet_loss[vidx]
                    # distribution unit do not provide reward

            # sku product unit balance sheet
            for pid in facility.product_unit_id_list:
                facility_balance_sheet_profit[i] += product_balance_sheet_profit[self.product_id2index_dict[pid]]
                facility_balance_sheet_loss[i] += product_balance_sheet_loss[self.product_id2index_dict[pid]]
                facility_step_reward[i] += product_step_reward[self.product_id2index_dict[pid]]

        facility_balance_sheet = facility_balance_sheet_loss + facility_balance_sheet_profit

        return facility_balance_sheet_profit, facility_balance_sheet_loss, facility_step_reward, facility_balance_sheet

    def calc(self):
        #### Basic Units: Loss, Profit, Reward
        consumer_ids, consumer_step_balance_sheet_loss, consumer_step_reward, consumer_step_balance_sheet = self._calc_consumer()
        seller_balance_sheet_profit, seller_balance_sheet_loss, seller_step_reward = self._calc_seller()
        manufacture_ids, manufacture_balance_sheet_loss, manufacture_step_reward, manufacture_step_balance_sheet = self._calc_manufacture()
        storage_balance_sheet_loss, storages_product_map = self._calc_storage()
        vehicle_balance_sheet_loss, vehicle_step_reward = self._calc_vehicle()
        product_distribution_balance_sheet_profit, product_distribution_balance_sheet_loss = self._calc_product_distribution()
        ########################################################################

        #### Loss, profit, reward for each product
        product_balance_sheet_profit, product_balance_sheet_loss, product_step_reward, product_balance_sheet = self._calc_product(
            consumer_step_balance_sheet_loss,
            consumer_step_reward,
            seller_balance_sheet_profit,
            seller_balance_sheet_loss,
            seller_step_reward,
            manufacture_balance_sheet_loss,
            manufacture_step_reward,
            storages_product_map,
            product_distribution_balance_sheet_profit,
            product_distribution_balance_sheet_loss
        )
        ########################################################################

        #### Loss, profit, reward for each facility
        # facility_balance_sheet_profit, facility_balance_sheet_loss, facility_step_reward, facility_balance_sheet = self._calc_facility(
        #     storage_balance_sheet_loss,
        #     vehicle_balance_sheet_loss,
        #     product_balance_sheet_profit,
        #     product_balance_sheet_loss,
        #     product_step_reward
        # )
        ########################################################################

        # Final result for current tick, key is the facility/unit id, value is tuple of balance sheet and reward.
        result = {}

        # For product units.
        for id, bs, rw in zip([product.unit_id for product in self.products], product_balance_sheet, product_step_reward):
            result[id] = (bs, rw)
            self.total_balance_sheet[id] += bs

        # For consumers.
        for id, bs, rw in zip(consumer_ids, consumer_step_balance_sheet, consumer_step_reward):
            # result[id] = (bs, rw)
            # let reward of a consumer equate its parent product
            result[id] = result[self.consumer_id2product[id]]
            self.total_balance_sheet[id] += result[id][0]

        # For producers.
        for id, bs, rw in zip(manufacture_ids, manufacture_step_balance_sheet, manufacture_step_reward):
            result[id] = (bs, rw)
            self.total_balance_sheet[id] += bs

        # NOTE: add followings if you need.
        # For storages.
        # For distributions.
        # For vehicles.

        return result

AGENT_IDS = [f"{info.agent_type}.{info.id}" for info in Env(**env_conf).agent_idx_list]
consumers = [agent_id for agent_id in AGENT_IDS if agent_id.startswith("consumer")]
agent2policy = {
    agent_id: agent_id.split(".")[0] for agent_id in AGENT_IDS if not agent_id.startswith("consumer")
}

for i, agent_id in enumerate(consumers):
    agent2policy[agent_id] = f"consumer-{i % NUM_RL_POLICIES}"


def get_env_sampler():
    return SCEnvSampler(
        lambda: Env(**env_conf),
        {**or_policy_func_dict, **policy_func_dict},
        agent2policy,
    )
