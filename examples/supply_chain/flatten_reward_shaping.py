from maro.simulator import Env
from collections import defaultdict, namedtuple
import scipy.stats as st
import numpy as np
from maro.rl import AbsEnvWrapper
from maro.simulator.scenarios.supply_chain.actions import ConsumerAction, ManufactureAction


class FlattenRewardShaping:
    """Old but fast reward shaping, without using snapshot wrapper"""
    consumer_features = ("id", "order_quantity", "price",
                         "order_cost", "order_product_cost", "reward_discount")
    seller_features = ("id", "sold", "demand", "price", "backlog_ratio")
    manufacture_features = ("id", "manufacturing_number", "product_unit_cost")
    product_features = (
        "id", "price", "distribution_check_order", "distribution_transport_cost", "distribution_delay_order_penalty")
    storage_features = ("capacity", "remaining_space")
    vehicle_features = ("id", "payload", "unit_transport_cost")

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

                downstream_product_units = []
                downstreams = facility["downstreams"]

                if downstreams and len(downstreams) > 0 and product_id in downstreams:
                    for dfacility in downstreams[product_id]:
                        dproducts = self.facilities[dfacility]["units"]["products"]

                        downstream_product_units.append(dproducts[product_id]["id"])

                self.products.append((
                    product["id"],
                    product_id,
                    product["node_index"],
                    facility["units"]["storage"]["node_index"],
                    facility["units"]["storage"]["config"]["unit_storage_cost"],
                    distribution["node_index"] if distribution is not None else None,
                    downstream_product_units,
                    None if consumer is None else (
                        consumer["id"], consumer["node_index"]),
                    None if seller is None else (
                        seller["id"], seller["node_index"]),
                    None if manufacture is None else (
                        manufacture["id"], manufacture["node_index"]),
                ))

            self.facility_levels.append((
                facility_id,
                pid_list,
                facility["units"]["storage"]["node_index"],
                facility["units"]["storage"]["config"]["unit_storage_cost"],
                distribution["node_index"] if distribution is not None else None,
                [v["node_index"] for v in distribution["children"]
                 ] if distribution is not None else []
            ))

        # TODO: order products make sure calculate reward from downstream to upstream
        tmp_product_unit_dict = {}

        for product in self.products:
            tmp_product_unit_dict[product[0]] = product

        self._ordered_products = []

        tmp_stack = []

        for product in self.products:
            # skip if already being processed
            if tmp_product_unit_dict[product[0]] is None:
                continue

            for dproduct in product[6]:
                # push downstream id to stack
                tmp_stack.append(dproduct)

            # insert current product to list head
            self._ordered_products.insert(0, product)
            # mark it as processed
            tmp_product_unit_dict[product[0]] = None

            while len(tmp_stack) > 0:
                # process downstream of product unit in stack
                dproduct_unit_id = tmp_stack.pop()

                # if it was processed then ignore
                if tmp_product_unit_dict[dproduct_unit_id] is None:
                    continue

                # or extract it downstreams
                dproduct_unit = tmp_product_unit_dict[dproduct_unit_id]

                dproduct_downstreams = dproduct_unit[6]

                for dproduct in dproduct_downstreams:
                    tmp_stack.append(dproduct)

                # current unit in final list
                self._ordered_products.insert(0, dproduct_unit)
                tmp_product_unit_dict[dproduct_unit_id] = None

        self.total_balance_sheet = defaultdict(int)

        # tick -> (product unit id, sku id, manufacture number, manufacture cost, checkin order, delay penaty)
        self._supplier_reward_factors = {}

    def calc(self):
        tick = self.env.tick

        # consumer
        consumer_bs_states = self.consumer_ss[tick::self.consumer_features]\
            .flatten()\
            .reshape(-1, len(self.consumer_features))

        # quantity * price
        order_profit = consumer_bs_states[:, 1] * consumer_bs_states[:, 2]

        # balance_sheet_profit = 0
        # order_cost + order_product_cost
        consumer_step_balance_sheet_loss = -1 * (consumer_bs_states[:, 3] + consumer_bs_states[:, 4])

        # consumer step reward: balance sheet los + profile * discount
        consumer_step_reward = consumer_step_balance_sheet_loss + \
            order_profit * consumer_bs_states[:, 5]

        # seller
        seller_bs_states = self.seller_ss[tick::self.seller_features]\
            .flatten()\
            .reshape(-1, len(self.seller_features))

        # profit = sold * price
        seller_balance_sheet_profit = seller_bs_states[:, 1] * seller_bs_states[:, 3]

        # loss = demand * price * backlog_ratio
        seller_balance_sheet_loss = -1 * \
            seller_bs_states[:, 2] * \
            seller_bs_states[:, 3] * seller_bs_states[:, 4]

        # step reward = loss + profit
        seller_step_reward = seller_balance_sheet_loss + seller_balance_sheet_profit

        # manufacture
        man_bs_states = self.manufacture_ss[tick::self.manufacture_features]\
            .flatten()\
            .reshape(-1, len(self.manufacture_features))

        # loss = manufacture number * cost
        man_balance_sheet_profit_loss = -1 * \
            man_bs_states[:, 1] * man_bs_states[:, 2]

        # step reward = loss
        man_step_reward = man_balance_sheet_profit_loss

        # product
        product_bs_states = self.product_ss[tick::self.product_features]\
            .flatten()\
            .reshape(-1, len(self.product_features))

        # product distribution loss = check order + delay order penalty
        product_distribution_balance_sheet_loss = -1 * \
            (product_bs_states[:, 3] + product_bs_states[:, 4])

        # product distribution profit = check order * price
        product_distribution_balance_sheet_profit = product_bs_states[:, 2] * product_bs_states[:, 1]

        # result we need
        product_step_reward = np.zeros((len(self.products, )))
        product_balance_sheet_profit = np.zeros((len(self.products, )))
        product_balance_sheet_loss = np.zeros((len(self.products, )))

        # create product number mapping for storages
        storages_product_map = {}
        for storage_index in range(len(self.storage_ss)):
            product_list = self.storage_ss[tick:storage_index:"product_list"]\
                .flatten()\
                .astype(np.int)

            product_number = self.storage_ss[tick:storage_index:"product_number"]\
                .flatten()\
                .astype(np.int)

            storages_product_map[storage_index] = {
                pid: pnum for pid, pnum in zip(product_list, product_number)
            }

        # product balance sheet and reward
        # loss = consumer loss + seller loss + manufacture loss + storage loss + distribution loss + downstreams loss
        # profit = same as above
        # reward = same as above
        for product in self._ordered_products:
            id, product_id, i, storage_index, unit_storage_cost, distribution_index, downstreams, consumer, seller, manufacture = product

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

            storage_reward = -1 * \
                storages_product_map[storage_index][product_id] * \
                unit_storage_cost

            product_step_reward[i] += storage_reward

            product_balance_sheet_loss[i] += storage_reward

            if distribution_index is not None:
                product_balance_sheet_loss[i] += product_distribution_balance_sheet_loss[i]
                product_balance_sheet_profit[i] += product_distribution_balance_sheet_profit[i]

                product_step_reward[i] += product_distribution_balance_sheet_loss[i] + \
                    product_distribution_balance_sheet_profit[i]

            if len(downstreams) > 0:
                for did in downstreams:
                    product_balance_sheet_loss[i] += product_balance_sheet_loss[self.product_id2index_dict[did]]
                    product_balance_sheet_profit[i] += product_balance_sheet_profit[self.product_id2index_dict[did]]
                    product_step_reward[i] += product_step_reward[self.product_id2index_dict[did]]

        product_balance_sheet = product_balance_sheet_profit + product_balance_sheet_loss

        # storage
        storage_states = self.storage_ss[tick::self.storage_features]\
            .flatten()\
            .reshape(-1, len(self.storage_features))

        # loss = (capacity-remaining space) * cost
        storage_balance_sheet_loss = -1 * (storage_states[:, 0] - storage_states[:, 1])
        vehicle_features = ("id", "payload", "unit_transport_cost")

        # vehicles
        vehicle_states = self.vehicle_ss[tick::self.vehicle_features]\
            .flatten()\
            .reshape(-1, len(self.vehicle_features))

        # loss = cost * payload
        vehicle_balance_sheet_loss = -1 * \
            vehicle_states[:, 1] * vehicle_states[:, 2]

        vehicle_step_reward = vehicle_balance_sheet_loss

        facility_balance_sheet_loss = np.zeros((len(self.facility_levels),))
        facility_balance_sheet_profit = np.zeros((len(self.facility_levels),))
        facility_step_reward = np.zeros((len(self.facility_levels),))

        # for facilities
        for i, facility in enumerate(self.facility_levels):
            id, pid_list, storage_index, unit_storage_cost, distribution_index, vehicle_indices = facility

            # storage balance sheet
            # profit=0
            facility_balance_sheet_loss[i] += storage_balance_sheet_loss[storage_index] * \
                unit_storage_cost

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

        # Final result for current tick, key is the facility/unit id, value is tuple of balance sheet and reward.
        result = {}

        # For product units.
        for id, bs, rw in zip([item[0] for item in self.products], product_balance_sheet, product_step_reward):
            result[id] = (bs, rw)

            self.total_balance_sheet[id] += bs

        facility_balance_sheet = facility_balance_sheet_loss + facility_balance_sheet_profit


        # For consumers.
        consumer_step_balance_sheet = order_profit + consumer_step_balance_sheet_loss

        for id, bs, rw in zip(consumer_bs_states[:, 0], consumer_step_balance_sheet, consumer_step_reward):
            result[int(id)] = (bs, rw)

            self.total_balance_sheet[id] += bs

        # For producers.
        man_step_balance_sheet = man_balance_sheet_profit_loss

        for id, bs, rw in zip(man_bs_states[:, 0], man_step_balance_sheet, man_step_reward):
            result[int(id)] = (bs, rw)

            self.total_balance_sheet[id] += bs

        # NOTE: add followings if you need.
        # For storages.
        # For distributions.
        # For vehicles.

        return result
