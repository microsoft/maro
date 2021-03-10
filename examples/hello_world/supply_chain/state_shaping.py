
"""
NOTE: this is used as a state shaping example, without maro.rl

states we used:

. is_positive_balance


is_over_stock
is_out_of_stock
is_below_rop
echelon_level

sale_std
storage_capacity
storage_utilization
sale_hist
consumption_hist
total_backlog_demand
inventory_in_stock
inventory_in_distribution
inventory_in_transit
inventory_estimated
inventory_rop

sku_price
sku_cost

"""

import numpy as np

from maro.simulator import Env


# NOTE: copied from original code
class BalanceSheet:
    profit: int = 0
    loss: int = 0

    def __init__(self, profit:int, loss:int):
        self.profit = profit
        self.loss = loss

    def total(self) -> int:
        return self.profit + self.loss

    def __add__(self, other):
        return BalanceSheet(self.profit + other.profit, self.loss + other.loss)

    def __sub__(self, other):
        return BalanceSheet(self.profit - other.profit, self.loss - other.loss)

    def __repr__(self):
        return f"{round(self.profit + self.loss, 0)} ({round(self.profit, 0)} {round(self.loss, 0)})"

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)


class SupplyChainStateShaping:
    def __init__(self, env: Env):
        self._env = env

    def shape(self):
        is_positive_balance = self._origin_group_by_sku_is_positive_balance()

        print(is_positive_balance)

    def _origin_group_by_sku_is_positive_balance(self):
        # original code collect states of facilities and sku related units (manufacture, seller, consumer),
        # other's states is added to facility
        result = []

        filter = lambda x: 1 if x > 1 else 0

        cur_frame_index = self._env.frame_index

        features = ("id", "facility_id", "balance_sheet_profit", "balance_sheet_loss")
        sku_related_features = features + ("product_id",)

        # facility balance sheet
        facility_nodes = self._env.snapshot_list["facility"]
        facility_balance_sheet = facility_nodes[cur_frame_index::features]
        facility_balance_sheet = facility_balance_sheet.flatten().reshape(len(facility_nodes), -1).astype(np.int)

        facility_balance_sheet_total = facility_balance_sheet[:, 2] + facility_balance_sheet[:, 3]

        facility_is_positive_balance = list(map(filter, facility_balance_sheet_total))

        result.extend(facility_is_positive_balance)

        # then each sku group for each facility
        manufacture_nodes = self._env.snapshot_list["manufacture"]
        seller_nodes = self._env.snapshot_list["seller"]
        consumer_nodes = self._env.snapshot_list["consumer"]

        manufacture_balance_sheet = manufacture_nodes[cur_frame_index::sku_related_features].flatten().reshape(len(manufacture_nodes), -1).astype(np.int)
        seller_balance_sheet = seller_nodes[cur_frame_index::sku_related_features].flatten().reshape(len(seller_nodes), -1).astype(np.int)
        consumer_balance_sheet = consumer_nodes[cur_frame_index::sku_related_features].flatten().reshape(len(consumer_nodes), -1).astype(np.int)

        #
        for facility_id in facility_balance_sheet[:, 0]:
            manufacture_states = manufacture_balance_sheet[manufacture_balance_sheet[:, 1] == facility_id]
            seller_states = seller_balance_sheet[seller_balance_sheet[:, 1] == facility_id]
            consumer_states = consumer_balance_sheet[consumer_balance_sheet[:, 1] == facility_id]

            length = max(len(manufacture_states), len(seller_states), len(consumer_states))

            # this facility does not have any sku related units
            if length == 0:
                continue

            sku_balance_sheet_total = np.zeros((length, ), dtype=np.int)

            if len(manufacture_states) > 0:
                sku_balance_sheet_total += manufacture_states[:, 2] + manufacture_states[:, 3]

            if len(seller_states) > 0:
                sku_balance_sheet_total += seller_states[:, 2] + seller_states[:, 3]

            if len(consumer_states) > 0:
                sku_balance_sheet_total += consumer_states[:, 2] + consumer_states[:, 3]

            sku_is_positive_balance = list(map(filter, sku_balance_sheet_total))

            result.extend(sku_is_positive_balance)

        return result
