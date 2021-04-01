# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator import Env
from collections import defaultdict, namedtuple
import scipy.stats as st
import numpy as np

from maro.simulator.scenarios.supply_chain.actions import ConsumerAction, ManufactureAction


def stock_constraint(f_state):
    return (0 < f_state['inventory_in_stock'] <= (f_state['max_vlt']+7)*f_state['sale_mean'])


def is_replenish_constraint(f_state):
    return (f_state['consumption_hist'][-1] > 0)


def low_profit(f_state):
    return ((f_state['sku_price']-f_state['sku_cost']) * f_state['sale_mean'] <= 1000)


def low_stock_constraint(f_state):
    return (0 < f_state['inventory_in_stock'] <= (f_state['max_vlt']+3)*f_state['sale_mean'])


def out_of_stock(f_state):
    return (0 < f_state['inventory_in_stock'])


atoms = {
    'stock_constraint': stock_constraint,
    'is_replenish_constraint': is_replenish_constraint,
    'low_profit': low_profit,
    'low_stock_constraint': low_stock_constraint,
    'out_of_stock': out_of_stock
}


class UnitBaseInfo:
    id: int = None
    node_index: int = None
    config: dict = None
    summary: dict = None

    def __init__(self, unit_summary):
        self.id = unit_summary["id"]
        self.node_index = unit_summary["node_index"]
        self.config = unit_summary.get("config", {})
        self.summary = unit_summary

    def __getitem__(self, key, default=None):
        if key in self.summary:
            return self.summary[key]

        return default


class SCEnvWrapper:
    def __init__(self, env: Env):
        self.env = env
        self.balance_cal = BalanceSheetCalculator(env)
        self.cur_balance_sheet_reward = None
        self.storage_ss = env.snapshot_list["storage"]

        self._summary = env.summary['node_mapping']
        self._configs = env.configs
        self._agent_types = self._summary["agent_types"]
        self._units_mapping = self._summary["unit_mapping"]
        self._agent_list = env.agent_idx_list

        self._sku_number = len(self._summary["skus"]) + 1

        # state for each tick
        self._cur_metrics = env.metrics
        self._cur_facility_storage_products = None

        # TODO: this is fixed after env setup
        self._cur_facility_storage_product_mapping = {}

        self._max_sources_per_facility = self._cur_metrics["max_sources_per_facility"]

        # facility -> {
        # data_model_index:int,
        # storage:UnitBaseInfo,
        # distribution: UnitBaseInfo,
        # product_id: {
        # consumer: UnitBaseInfo,
        # seller: UnitBaseInfo,
        # manufacture: UnitBaseInfo
        # }
        # }
        self.facility_levels = {}

        # unit id -> (facility id)
        self.unit_2_facility_dict = {}

        for facility_id, facility in self._summary["facilities"].items():
            self.facility_levels[facility_id] = {
                "node_index": facility["node_index"],
                "config": facility['configs'],
                "upstreams": facility["upstreams"],
                "skus": facility["skus"]
            }

            units = facility["units"]

            storage = units["storage"]
            if storage is not None:
                self.facility_levels[facility_id]["storage"] = UnitBaseInfo(storage)

                self.unit_2_facility_dict[storage["id"]] = facility_id

            distribution = units["distribution"]

            if distribution is not None:
                self.facility_levels[facility_id]["distribution"] = UnitBaseInfo(distribution)
                self.unit_2_facility_dict[distribution["id"]] = facility_id

            products = units["products"]

            if products:
                for product_id, product in products.items():
                    product_info = {
                        "skuproduct": UnitBaseInfo(product)
                    }

                    self.unit_2_facility_dict[product["id"]] = facility_id

                    seller = product['seller']

                    if seller is not None:
                        product_info["seller"] = UnitBaseInfo(seller)
                        self.unit_2_facility_dict[seller["id"]] = facility_id

                    consumer = product["consumer"]

                    if consumer is not None:
                        product_info["consumer"] = UnitBaseInfo(consumer)
                        self.unit_2_facility_dict[consumer["id"]] = facility_id

                    manufacture = product["manufacture"]

                    if manufacture is not None:
                        product_info["manufacture"] = UnitBaseInfo(manufacture)
                        self.unit_2_facility_dict[manufacture["id"]
                                                  ] = facility_id

                    self.facility_levels[facility_id][product_id] = product_info

    def get_state(self, event):
        self.cur_balance_sheet_reward = self.balance_cal.calc()

        return self._get_state()

    def get_action(self, action_by_agent):
        env_action = {}
        for agent_id, action in action_by_agent.items():
            # consumer action
            if agent_id in self.state_info:
                pd_id, sr_id = self.state_info[agent_id]["product_id"], self.state_info[agent_id]["source_id"]
                env_action[agent_id] = ConsumerAction(
                    agent_id, pd_id, sr_id, action, 1)
            # manufacturer action
            else:
                env_action[agent_id] = ManufactureAction(agent_id, action)

        return env_action

    def get_reward(self, tick=None):
        wc = self.env.configs.settings["global_reward_weight_consumer"]

        parent_facility_balance = {}

        for f_id, sheet in step_balance_sheet.items():
            if f_id in self.unit_2_facility_dict:
                # it is a product unit
                parent_facility_balance[f_id] = step_balance_sheet[self.unit_2_facility_dict[f_id]]
            else:
                parent_facility_balance[f_id] = sheet

        consumer_reward_by_facility = { f_id: wc * parent_facility_balance[f_id] + (1 - wc) * bsw[1] for f_id, bsw in self.cur_balance_sheet_reward.items() }

        rewards_by_agent = {
            "consumer": {},
            "producer": {}
        }

        for f_id, bsw in self.cur_balance_sheet_reward.items():
            rewards_by_agent["producer"][f_id] = bsw[1]

        for f_id, reward in consumer_reward_by_facility.items():
            rewards_by_agent["consumer"][f_id] = reward

        return rewards_by_agent

    def _get_state(self):
        self._cur_metrics = self.env.metrics

        state = {
            "consumer": {},
            "producer": {}
        }

        for agent_info in self._agent_list:
            storage_index = self.facility_levels[agent_info.facility_id]['storage'].node_index

            storage_product_list = self.storage_ss[self.env.tick:storage_index:"product_list"].flatten().astype(np.int)
            storage_product_number = self.storage_ss[self.env.tick:storage_index:"product_number"].flatten().astype(np.int)

            self._cur_facility_storage_products = { pid:pnum for pid, pnum in  zip(storage_product_list, storage_product_number)}

            self._cur_facility_storage_product_mapping = {product_id: i for i, product_id in enumerate(storage_product_list)}

            f_state = self._state(agent_info)

            self._add_global_features(f_state)

            state['consumer'][agent_info.id] = f_state
            state['producer'][agent_info.id] = f_state

        return self._serialize_state(state)

    def _state(self, agent_info):
        state = {}

        self._add_facility_features(state, agent_info)
        self._add_storage_features(state, agent_info)
        self._add_bom_features(state, agent_info)
        self._add_distributor_features(state, agent_info)
        self._add_sale_features(state, agent_info)
        self._add_vlt_features(state, agent_info)
        self._add_consumer_features(state, agent_info)
        self._add_price_features(state, agent_info)

        return state

    def _add_global_features(self, state):
        state['global_time'] = self.env.tick

    def _add_facility_features(self, state, agent_info):
        state['facility_type'] = [
            1 if i == agent_info.agent_type else 0 for i in range(len(self._agent_types))]

        # NOTE: We cannot provide facility instance
        state["facility"] = None

        state["is_accepted"] = [0] * self._configs.settings["constraint_state_hist_len"]

        # NOTE: we have no constraint now
        state['constraint_idx'] = [0]

        for atom_name in atoms.keys():
            state[atom_name] = list(np.ones(self._configs.settings['constraint_state_hist_len']))

        # NOTE: named as facility_id but actually sku id
        # NOTE: as our sku id start from 1, so we need expend one-hot length
        state['facility_id'] = [0] * self._sku_number

        facility = self.facility_levels[agent_info.facility_id]

        if agent_info.is_facility:
            # truely facility
            state['facility_info'] = facility['config']
            state['sku_info'] = {}

            metrics = self._cur_metrics["facilities"][agent_info.facility_id]

            state['is_positive_balance'] = 1 if self.balance_cal.total_balance_sheet[agent_info.id] > 0 else 0
        else:
            # a product unit
            # 3rd slot is the facility id of this unit
            state['facility_info'] = facility['config']
            state['sku_info'] = agent_info.sku

            metrics = self._cur_metrics["products"][agent_info.id]
            state['is_positive_balance'] = 1 if self.balance_cal.total_balance_sheet[agent_info.id] > 0 else 0

            # NOTE: ignore constraint here

            state['facility_id'][agent_info.sku.id] = 1

            # NOTE: ignore atom here

        # NOTE: ignore this as we do not implement it now
        state['echelon_level'] = 0

    def _add_storage_features(self, state, agent_info):
        facility = self.facility_levels[agent_info.facility_id]

        state['storage_levels'] = [0] * self._sku_number

        state['storage_capacity'] = facility['storage'].config["capacity"]

        state['storage_utilization'] = 0

        for product_id, product_number in self._cur_facility_storage_products.items():
            state['storage_levels'][product_id] = product_number
            state['storage_utilization'] += product_number

    def _add_bom_features(self, state, agent_info):
        state['bom_inputs'] = [0] * self._sku_number
        state['bom_outputs'] = [0] * self._sku_number

        if not agent_info.is_facility:
            state['bom_inputs'][agent_info.sku.id] = 1
            state['bom_outputs'][agent_info.sku.id] = 1

    def _add_vlt_features(self, state, agent_info):
        sku_list = self._summary["skus"]
        facility = self.facility_levels[agent_info.facility_id]

        current_source_list = []

        # only for product unit
        if agent_info.sku is not None:
            current_source_list = facility["upstreams"].get(agent_info.sku.id, [])

        state['vlt'] = [0] * (self._max_sources_per_facility * self._sku_number)
        state['max_vlt'] = 0

        if not agent_info.is_facility:
            # only for sku product
            product_info = facility[agent_info.sku.id]

            if "consumer" in product_info and len(current_source_list) > 0:
                state['max_vlt'] = product_info["skuproduct"]["max_vlt"]

                for i, source in enumerate(current_source_list):
                    for j, sku in enumerate(sku_list.values()):
                        # NOTE: different with original code, our config can make sure that source has product we need

                        if sku.id == agent_info.sku.id:
                            state['vlt'][i * len(sku_list) + j + 1] = facility["skus"][sku.id].vlt

    def _add_sale_features(self, state, agent_info):
        state['sale_mean'] = 1.0
        state['sale_std'] = 1.0
        state['sale_gamma'] = 1.0
        state['service_level'] = 0.95
        state['total_backlog_demand'] = 0

        settings = self.env.configs.settings

        hist_len = settings['sale_hist_len']
        consumption_hist_len = settings['consumption_hist_len']

        state['sale_hist'] = [0] * hist_len
        state['backlog_demand_hist'] = [0] * hist_len
        state['consumption_hist'] = [0] * consumption_hist_len
        state['pending_order'] = [0] * settings['pending_order_len']

        if agent_info.is_facility:
            return

        product_metrics = self._cur_metrics["products"][agent_info.id]

        # for product unit only
        state['service_level'] = agent_info.sku.service_level
        state['sale_mean'] = product_metrics["sale_mean"]
        state['sale_gamma'] = state['sale_mean']
        state['sale_std'] = product_metrics["sale_std"]

        facility = self.facility_levels[agent_info.facility_id]
        product_info = facility[agent_info.sku.id]

        if "consumer" in product_info:
            # TODO: implement later
            consumer_index = product_info["consumer"].node_index

            consumption_hist = self.env.snapshot_list["consumer"][[self.env.tick - i for i in range(consumption_hist_len)]:consumer_index:"latest_consumptions"]
            consumption_hist = consumption_hist.flatten()

            state['consumption_hist'] = list(consumption_hist)
            state['pending_order'] = list(product_metrics["pending_order_daily"])

        if "seller" in product_info:
            seller_index = product_info["seller"].node_index
            seller_ss = self.env.snapshot_list["seller"]

            single_states = seller_ss[self.env.tick:seller_index:("total_demand")].flatten().astype(np.int)
            hist_states = seller_ss[[self.env.tick - i for i in range(hist_len)]:seller_index:("sold", "demand")].flatten().reshape(2, -1).astype(np.int)

            state['total_backlog_demand'] = single_states[0]
            state['sale_hist'] = list(hist_states[0])
            state['backlog_demand_hist'] = list(hist_states[1])
            state['sale_gamma'] = facility["skus"][agent_info.sku.id].sale_gamma

    def _add_distributor_features(self, state, agent_info):
        state['distributor_in_transit_orders'] = 0
        state['distributor_in_transit_orders_qty'] = 0

        facility = self.facility_levels[agent_info.facility_id]

        distribution = facility.get("distribution", None)

        if distribution is not None:
            dist_states = self.env.snapshot_list["distribution"][self.env.tick:distribution.id:("remaining_order_quantity", "remaining_order_number")]
            dist_states = dist_states.flatten().astype(np.int)

            state['distributor_in_transit_orders'] = dist_states[1]
            state['distributor_in_transit_orders_qty'] = dist_states[0]

    def _add_consumer_features(self, state, agent_info):
        state['consumer_source_export_mask'] = [0] * (self._max_sources_per_facility * self._sku_number)
        state['consumer_source_inventory'] = [0] * self._sku_number
        state['consumer_in_transit_orders'] = [0] * self._sku_number

        state['inventory_in_stock'] = 0
        state['inventory_in_transit'] = 0
        state['inventory_in_distribution'] = 0
        state['inventory_estimated'] = 0
        state['inventory_rop'] = 0
        state['is_over_stock'] = 0
        state['is_out_of_stock'] = 0
        state['is_below_rop'] = 0

        if agent_info.is_facility:
            return

        facility = self.facility_levels[agent_info.facility_id]
        product_info = facility[agent_info.sku.id]

        if "consumer" not in product_info:
            return

        source_list = facility["upstreams"].get(agent_info.sku.id, [])

        if len(source_list) == 0:
            return

        sku_list = self._summary["skus"]

        for i, source in enumerate(source_list):
            for j, sku in enumerate(sku_list.values()):
                if sku.id == agent_info.sku.id:
                    state['consumer_source_export_mask'][i * len(sku_list) + j + 1] = self.facility_levels[source]["skus"][sku.id].vlt

        in_transit_orders = self._cur_metrics['facilities'][agent_info.facility_id]["in_transit_orders"]

        for i, sku in enumerate(sku_list.values()):
            state['consumer_in_transit_orders'][sku.id] += in_transit_orders[sku.id]

        state['inventory_in_stock'] = self._cur_facility_storage_products[agent_info.sku.id]
        state['inventory_in_transit'] = state['consumer_in_transit_orders'][agent_info.sku.id]

        pending_order = self._cur_metrics["facilities"][agent_info.facility_id]["pending_order"]

        if pending_order is not None:
            state['inventory_in_distribution'] = pending_order[agent_info.sku.id]

        state['inventory_estimated'] = (state['inventory_in_stock']
                                         + state['inventory_in_transit']
                                         - state['inventory_in_distribution'])
        if (state['inventory_estimated'] >= 0.5*state['storage_capacity']):
            state['is_over_stock'] = 1

        if (state['inventory_estimated'] <= 0):
            state['is_out_of_stock'] = 1

        state['inventory_rop'] = (state['max_vlt']*state['sale_mean']
                                  + np.sqrt(state['max_vlt'])*state['sale_std']*st.norm.ppf(state['service_level']))

        if state['inventory_estimated'] < state['inventory_rop']:
            state['is_below_rop'] = 1

    def _add_price_features(self, state, agent_info):
        state['max_price'] = self._cur_metrics["max_price"]
        state['sku_price'] = 0
        state['sku_cost'] = 0

        if not agent_info.is_facility:
            state['sku_price'] = agent_info.sku.price
            state['sku_cost'] = agent_info.sku.cost

    def _serialize_state(self, state):
        result = {
            "consumer": {},
            "producer": {}
        }

        keys_in_state = [(None, ['is_over_stock', 'is_out_of_stock', 'is_below_rop',
                                 'constraint_idx', 'is_accepted', 'consumption_hist']),
                         ('storage_capacity', ['storage_utilization']),
                         ('sale_gamma', ['sale_std',
                                         'sale_hist',
                                         'pending_order',
                                         'inventory_in_stock',
                                         'inventory_in_transit',
                                         'inventory_estimated',
                                         'inventory_rop']),
                         ('max_price', ['sku_price', 'sku_cost'])]

        for _type, agents_dict in state.items():
            for agent_id, agent_raw_state in agents_dict.items():
                result[_type][agent_id] = []

                for norm, fields in keys_in_state:
                    for field in fields:
                        vals = agent_raw_state[field]

                        if not isinstance(vals, list):
                            vals = [vals]
                        if norm is not None:
                            vals = [
                                max(0.0, min(100.0, x/(agent_raw_state[norm]+0.01))) for x in vals]

                        result[_type][agent_id].extend(vals)
                result[_type][agent_id] = np.array(result[_type][agent_id])

        return result

class BalanceSheetCalculator:
    consumer_features = ("id", "order_quantity", "price", "order_cost", "order_product_cost")
    seller_features = ("id", "sold", "demand", "price", "backlog_ratio")
    manufacture_features = ("id", "manufacturing_number", "product_unit_cost")
    product_features = ("id", "price", "distribution_check_order", "distribution_transport_cost", "distribution_delay_order_penalty")
    storage_features = ("capacity", "remaining_space")
    vehicle_features=  ("id", "payload", "unit_transport_cost")

    def __init__(self, env: Env):
        self.env = env
        self.consumer_ss = env.snapshot_list["consumer"]
        self.seller_ss = env.snapshot_list["seller"]
        self.manufacture_ss = env.snapshot_list["manufacture"]
        self.storage_ss = env.snapshot_list["storage"]
        self.distribution_ss = env.snapshot_list["distribution"]
        self.vehicle_ss = env.snapshot_list["vehicle"]
        self.product_ss = env.snapshot_list["product"]
        self.products = []
        self.product_id2index_dict = {}
        self.facility_levels = []

        self.facilities = env.summary["node_mapping"]["facilities"]

        for facility_id, facility in self.facilities.items():
            pid_list = []
            distribution = facility["units"]["distribution"]

            for product_id, product in facility["units"]["products"].items():
                pid_list.append(product["id"])
                consumer = product["consumer"]
                seller = product["seller"]
                manufacture = product["manufacture"]

                self.product_id2index_dict[product["id"]] = len(self.products)

                self.products.append((
                    product["id"],
                    product_id,
                    facility["units"]["storage"]["node_index"],
                    facility["units"]["storage"]["config"]["unit_storage_cost"],
                    distribution["node_index"] if distribution is not None else None,
                    facility["downstreams"],
                    None if consumer is None else (consumer["id"], consumer["node_index"]),
                    None if seller is None else (seller["id"], seller["node_index"]),
                    None if manufacture is None else (manufacture["id"], manufacture["node_index"]),
                ))

            self.facility_levels.append((
                facility_id,
                pid_list,
                facility["units"]["storage"]["node_index"],
                facility["units"]["storage"]["config"]["unit_storage_cost"],
                distribution["node_index"] if distribution is not None else None,
                [v["node_index"] for v in distribution["children"]] if distribution is not None else []
            ))

        self.total_balance_sheet = defaultdict(int)

    def calc(self):
        tick = self.env.tick
        # consumer
        consumer_bs_states = self.consumer_ss[tick::self.consumer_features].flatten().reshape(-1, len(self.consumer_features))

        # quantity * price
        consumer_profit = consumer_bs_states[:, 1] * consumer_bs_states[:, 2]

        # balance_sheet_profit = 0
        # order_cost + order_product_cost
        consumer_step_balance_sheet_loss = -1 * (consumer_bs_states[:, 3] + consumer_bs_states[:, 4])

        # not sure about this.
        reward_discount = 0

        # consumer step reward: balance sheet los + profile * discount
        consumer_step_reward = consumer_step_balance_sheet_loss + consumer_profit * reward_discount

        # seller
        seller_bs_states = self.seller_ss[tick::self.seller_features].flatten().reshape(-1, len(self.seller_features))

        # profit = sold * price
        seller_balance_sheet_profit = seller_bs_states[:, 1] * seller_bs_states[:, 3]

        # loss = demand * price * backlog_ratio
        seller_balance_sheet_loss = -1 * seller_bs_states[:, 2] * seller_bs_states[:, 3] * seller_bs_states[:, 4]

        # step reward = loss + profit
        seller_step_reward = seller_balance_sheet_loss + seller_balance_sheet_profit

        # manufacture
        man_bs_states = self.manufacture_ss[tick::self.manufacture_features].flatten().reshape(-1, len(self.manufacture_features))

        # loss = manufacture number * cost
        man_balance_sheet_profit_loss = -1 * man_bs_states[:, 1] * man_bs_states[:, 2]

        # step reward = loss
        man_step_reward = man_balance_sheet_profit_loss

        # product
        product_bs_states = self.product_ss[tick::self.product_features].flatten().reshape(-1, len(self.product_features))

        # product distribution loss = check order + delay order penalty
        product_distribution_balance_sheet_loss= -1 * (product_bs_states[:, 3] + product_bs_states[:, 4])

        # product distribution profit = check order * price
        product_distribution_balance_sheet_profit = product_bs_states[:, 2] * product_bs_states[:, 1]

        # result we need 
        product_step_reward = np.zeros((len(self.products,)))
        product_balance_sheet_profit = np.zeros((len(self.products,)))
        product_balance_sheet_loss = np.zeros((len(self.products, )))

        # create product number mapping for storages
        storages_product_map = {}
        for storage_index in range(len(self.storage_ss)):
            product_list = self.storage_ss[tick:storage_index:"product_list"].flatten().astype(np.int)
            product_number = self.storage_ss[tick:storage_index:"product_number"].flatten().astype(np.int)

            storages_product_map[storage_index] = {pid:pnum for pid, pnum in zip(product_list, product_number)}

        # product balance sheet and reward
        # loss = consumer loss + seller loss + manufacture loss + storage loss + distribution loss + downstreams loss
        # profit = same as above
        # reward = same as above
        for i, product in enumerate(self.products):
            id, product_id, storage_index, unit_storage_cost, distribution_index, downstreams, consumer, seller, manufacture = product

            if consumer:
                product_balance_sheet_loss[i] += consumer_step_balance_sheet_loss[consumer[1]]
                product_step_reward[i] += consumer_step_reward[consumer[1]]

            if seller:
                product_balance_sheet_loss[i] += seller_balance_sheet_loss[seller[1]]
                product_balance_sheet_profit[i] += seller_balance_sheet_profit[seller[1]]
                product_step_reward[i] += seller_step_reward[seller[1]]

            if manufacture:
                product_balance_sheet_loss[i] += man_balance_sheet_profit_loss[manufacture[1]]
                product_step_reward[i] += man_step_reward[manufacture[1]]

            storage_reward = -1 * storages_product_map[storage_index][product_id] * unit_storage_cost

            product_step_reward[i] += storage_reward

            product_balance_sheet_loss[i] += storage_reward

            if distribution_index is not None:
                product_balance_sheet_loss[i] += product_distribution_balance_sheet_loss[distribution_index]
                product_balance_sheet_profit[i] += product_distribution_balance_sheet_profit[distribution_index]

                product_step_reward[i] += product_distribution_balance_sheet_loss[distribution_index] + product_distribution_balance_sheet_profit[distribution_index]

            if downstreams and len(downstreams) > 0:
                if product_id in downstreams:
                    for dfacility in downstreams[product_id]:
                        dproducts = self.facilities[dfacility]["units"]["products"]

                        did = dproducts[product_id]["id"]

                        product_balance_sheet_loss[i] += product_balance_sheet_loss[self.product_id2index_dict[did]]
                        product_balance_sheet_profit[i] += product_balance_sheet_profit[self.product_id2index_dict[did]]
                        product_step_reward[i] += product_step_reward[self.product_id2index_dict[did]]

        product_balance_sheet = product_balance_sheet_profit + product_balance_sheet_loss

        # storage
        storage_states = self.storage_ss[tick::self.storage_features].flatten().reshape(-1, len(self.storage_features))

        # loss = (capacity-remaining space) * cost 
        storage_balance_sheet_loss = -1 * (storage_states[:, 0] - storage_states[:, 1])

        # vehicles
        vehicle_states = self.vehicle_ss[tick::self.vehicle_features].flatten().reshape(-1, len(self.vehicle_features))

        # loss = cost * payload
        vehicle_balance_sheet_loss = -1 * vehicle_states[:, 1] * vehicle_states[:, 2]
        vehicle_step_reward = vehicle_balance_sheet_loss

        facility_balance_sheet_loss = np.zeros((len(self.facility_levels),))
        facility_balance_sheet_profit = np.zeros((len(self.facility_levels),))
        facility_step_reward = np.zeros((len(self.facility_levels),))

        # for facilities
        for i, facility in enumerate(self.facility_levels):
            id, pid_list, storage_index, unit_storage_cost, distribution_index, vehicle_indices = facility

            # storage balance sheet
            # profit=0
            facility_balance_sheet_loss[i] += storage_balance_sheet_loss[storage_index] * unit_storage_cost

            # distribution balance sheet
            if distribution_index is not None:
                for vindex in vehicle_indices:
                    facility_balance_sheet_loss[i] += vehicle_balance_sheet_loss[vindex]
                    # distribution unit do not provide reward

            # sku product unit balance sheet
            for pid in pid_list:
                facility_balance_sheet_loss[i] += product_balance_sheet_loss[self.product_id2index_dict[pid]]
                facility_balance_sheet_profit[i] += product_balance_sheet_profit[self.product_id2index_dict[pid]]
                facility_step_reward[i] += product_step_reward[self.product_id2index_dict[pid]]

        result = {}

        for id, bs, rw in zip([item[0] for item in self.products], product_balance_sheet, product_step_reward):
            result[id] = (bs, rw)

            self.total_balance_sheet[id] += bs

        facility_balance_sheet = facility_balance_sheet_loss + facility_balance_sheet_profit

        for id, bs, rw in zip([item[0] for item in self.facility_levels], facility_balance_sheet, facility_step_reward):
            result[id] = (bs, rw)

            self.total_balance_sheet[id] += bs

        return result

if __name__ == "__main__":
    from time import time

    start_tick = 0
    durations = 100
    env = Env(scenario="supply_chain", topology="sample1",
              start_tick=start_tick, durations=durations)

    ss = SCEnvWrapper(env)

    env.step(None)

    # bbs = BalanceSheetCalculator(env)

    # start_time = time()
    # ss = bbs.calc()
    # end_time = time()

    # print(ss)

    # print("time cost: ", end_time - start_time)

    start_time = time()
    ss.get_state(None)
    # ss.get_reward(None)
    end_time = time()

    print("time cost: ", end_time - start_time)
    # print(env.metrics)
