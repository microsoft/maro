
from maro.simulator.snapshotwrapper import SnapshotWrapper, SnapshotRelationTree
from maro.simulator.snapshotwrapper.nodewrapper import SnapshotNodeWrapper
from typing import Tuple
from maro.simulator.core import AbsEnv
from collections import defaultdict


class BalanceSheet:
    profit = 0
    loss = 0

    def __init__(self, profit, loss):
        self.profit = profit
        self.loss = loss

    def sum(self):
        return self.profit + self.loss

    def __add__(self, other):
        assert type(other) == BalanceSheet

        return BalanceSheet(self.profit + other.profit, self.loss + other.loss)

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self):
        return f"<BalanceSheet profit: {self.profit}, loss: {self.loss}>"

    def __repr__(self):
        return self.__str__()


def bslist_sum(bs_list: list):
    if bs_list is None or len(bs_list) == 0:
        return BalanceSheet(0, 0)

    r = bs_list[0]

    for rr in bs_list[1:]:
        r += rr

    return r


BalancSheetReward = Tuple[BalanceSheet, float]


class RewardShaping:
    def __init__(self, env: AbsEnv):
        self.env = env

        self.total_balance_sheet = defaultdict(int)

        self._sswraper = SnapshotWrapper(self.env)
        self._update_facility_id_list = None

    def calc(self):
        self._sswraper.update()

        if self._update_facility_id_list is None:
            self._update_facility_id_list = self._sswraper.relations["downstreams"].down_to_up()

        bs_reward_dict = {}

        for facility_id in self._update_facility_id_list:
            facility_node = self._sswraper.get_node_by_id(facility_id)

            self._process_facility(bs_reward_dict, facility_node)

        result = {}

        for k, v in bs_reward_dict.items():
            bs_total = v[1].sum()

            if type(v[0]).__name__ in ("consumer", "manufacture", "product"):
                result[k] = (bs_total, v[2])

            self.total_balance_sheet[k] = bs_total

        return result

    def _process_facility(self, bsr_dict: dict, facility: SnapshotNodeWrapper) -> BalancSheetReward:
        facility_cls_name = type(facility).__name__

        if facility_cls_name == "supplier":
            return self._process_supplier_facility(bsr_dict, facility)
        elif facility_cls_name == "warehouse":
            return self._process_warehouse_facility(bsr_dict, facility)
        elif facility_cls_name == "retailer":
            return self._process_retailer_facility(bsr_dict, facility)
        else:
            raise Exception("Invalid facility type.")

    def _process_supplier_facility(self, bsr_dict: dict, facility: SnapshotNodeWrapper) -> BalancSheetReward:
        bs, reward = self._process_retailer_facility(bsr_dict, facility)

        bsr_dict[facility.maro_uid] = (facility, bs, reward)

        return bs, reward

    def _process_warehouse_facility(self, bsr_dict: dict, facility: SnapshotNodeWrapper) -> BalancSheetReward:
        bs, reward = self._process_retailer_facility(bsr_dict, facility)

        bsr_dict[facility.maro_uid] = (facility, bs, reward)

        return bs, reward

    def _process_retailer_facility(self, bsr_dict: dict, facility: SnapshotNodeWrapper) -> BalancSheetReward:
        bs_list = []
        reward_list = []

        # storage
        storage_unit = facility.children.storage

        storage_bs, storage_reward = self._process_storage_unit(bsr_dict, storage_unit)

        bs_list.append(BalanceSheet(0, storage_bs.loss * storage_unit.unit_storage_cost))

        # distribution
        dist_unit = getattr(facility.children, 'distribution', None)

        if dist_unit is not None:
            dist_bs, dist_reward = self._process_distribution_unit(bsr_dict, dist_unit)

            bs_list.append(dist_bs)

        # products
        products = facility.children.product

        if type(products) != list:
            products = [products]

        for product_unit in products:
            product_bs, product_reward = self._process_product_unit(bsr_dict, product_unit)

            bs_list.append(product_bs)
            reward_list.append(product_reward)

        bs, reward = bslist_sum(bs_list), sum(reward_list)

        bsr_dict[facility.maro_uid] = (facility, bs, reward)

        return bs, reward

    def _process_distribution_unit(self, bsr_dict: dict, unit: SnapshotNodeWrapper) -> BalancSheetReward:
        bs_list = []
        reward_list = []

        for vehicle in unit.children.vehicle:
            bs, reward = self._process_vehicle_unit(bsr_dict, vehicle)

            bs_list.append(bs)
            reward_list.append(reward)

        bs, reward = bslist_sum(bs_list), sum(reward_list)

        bsr_dict[unit.maro_uid] = (unit, bs, reward)

        return bs, reward

    def _process_vehicle_unit(self, bsr_dict: dict, unit: SnapshotNodeWrapper) -> BalancSheetReward:
        bs_loss = -(unit.unit_transport_cost * unit.payload)

        bs = BalanceSheet(0, bs_loss)

        reward = bs.sum()

        bsr_dict[unit.maro_uid] = (unit, bs, reward)

        return bs, reward

    def _process_storage_unit(self, bsr_dict: dict, unit: SnapshotNodeWrapper) -> BalancSheetReward:
        bs_loss = -(unit.capacity - unit.remaining_space)

        bs = BalanceSheet(0, bs_loss)
        reward = bs.sum()

        bsr_dict[unit.maro_uid] = (unit, bs, reward)

        return bs, reward

    def _process_product_unit(self, bsr_dict: dict, unit: SnapshotNodeWrapper) -> BalancSheetReward:
        bs_list = []
        reward_list = []

        for cname, process_func in (
            ('consumer', self._process_consumer_unit),
            ('seller', self._process_seller_unit),
            ('manufacture', self._process_manufacture_unit)):
            if hasattr(unit.children, cname):
                child = getattr(unit.children, cname)

                child_bs, child_reward = process_func(bsr_dict, child)

                if cname == 'consumer':
                    bs_list.append(BalanceSheet(0, child_bs.loss))
                else:
                    bs_list.append(child_bs)
                reward_list.append(child_reward)

        # distribution
        dist_bs_loss = -(unit.distribution_transport_cost + unit.distribution_delay_order_penalty)
        dist_bs_profit = unit.distribution_check_order * unit.price
        dist_bs = BalanceSheet(dist_bs_profit, dist_bs_loss)

        bs_list.append(dist_bs)
        reward_list.append(dist_bs.sum())

        # storage reward
        storage_unit = unit.maro_parent.children.storage

        product_list = storage_unit.product_list
        product_number = storage_unit.product_number

        idx = None

        for i, pid in enumerate(product_list):
            if pid == int(unit.product_id):
                idx = i
                break

        assert idx is not None

        storage_loss = -(product_number[idx] * storage_unit.unit_storage_cost)

        storage_bs = BalanceSheet(0, storage_loss)

        bs_list.append(storage_bs)
        reward_list.append(storage_loss)

        # NOTE: no downstream included.

        total_bs, total_reward = bslist_sum(bs_list), sum(reward_list)

        bsr_dict[unit.maro_uid] = (unit, total_bs, total_reward)

        return total_bs, total_reward

    def _process_manufacture_unit(self, bsr_dict: dict, unit: SnapshotNodeWrapper) -> BalancSheetReward:
        # manufacture unit has no profit
        bs_loss = -(unit.manufacturing_number * unit.product_unit_cost)

        bs = BalanceSheet(0, bs_loss)
        reward = bs.sum()

        bsr_dict[unit.maro_uid] = (unit, bs, reward)

        return bs, reward

    def _process_seller_unit(self, bsr_dict: dict, unit: SnapshotNodeWrapper) -> BalancSheetReward:
        bs_profit = unit.sold * unit.price
        bs_loss = -(unit.demand * unit.price * unit.backlog_ratio)

        bs = BalanceSheet(bs_profit, bs_loss)
        reward = bs.sum()

        bsr_dict[unit.maro_uid] = (unit, bs, reward)

        return bs, reward

    def _process_consumer_unit(self, bsr_dict: dict, unit: SnapshotNodeWrapper) -> BalancSheetReward:
        order_profit = unit.order_quantity * unit.price

        # consumer always spend money to buy, so no profit.
        bs_loss = -(unit.order_cost + unit.order_product_cost)

        bs = BalanceSheet(order_profit, bs_loss)

        reward = bs_loss + order_profit * unit.reward_discount

        bsr_dict[unit.maro_uid] = (unit, bs, reward)

        return bs, reward
