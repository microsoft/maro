
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
        self.is_positive_balance()

    def is_positive_balance(self):
        features = ("id", "facility_id", "balance_sheet_profit", "balance_sheet_loss")

        storage_nodes =  self._env.snapshot_list["storage"]
        storage_number = len(storage_nodes)

        storage_balance_sheet = storage_nodes[self._env.frame_index::features]
        storage_balance_sheet = storage_balance_sheet.flatten().reshape(storage_number, -1).astype(np.int)

        distribution_nodes = self._env.snapshot_list["distribution"]
        distribution_number = len(distribution_nodes)

        distribution_balance_sheet = distribution_nodes[self._env.frame_index::features]
        distribution_balance_sheet = distribution_balance_sheet.flatten().reshape(distribution_number, -1).astype(np.int)

        transport_nodes = self._env.snapshot_list["transport"]
        transport_number = len(transport_nodes)

        transport_balance_sheet = transport_nodes[self._env.frame_index::features]
        transport_balance_sheet = transport_balance_sheet.flatten().reshape(transport_number, -1).astype(np.int)

        manufacture_nodes = self._env.snapshot_list["manufacture"]
        manufacture_number = len(manufacture_nodes)

        manufacture_balance_sheet = manufacture_nodes[self._env.frame_index::features]
        manufacture_balance_sheet = manufacture_balance_sheet.flatten().reshape(manufacture_number, -1).astype(np.int)

        seller_nodes = self._env.snapshot_list["seller"]
        seller_number = len(seller_nodes)

        seller_balance_sheet = seller_nodes[self._env.frame_index::features]
        seller_balance_sheet = seller_balance_sheet.flatten().reshape(seller_number, -1).astype(np.int)

        facility_nodes = self._env.snapshot_list["facility"]
        facility_number = len(facility_nodes)

        facility_balance_sheet = facility_nodes[self._env.frame_index::features]
        facility_balance_sheet = facility_balance_sheet.flatten().reshape(facility_number, -1).astype(np.int)

        print("storage balance sheet")
        print(storage_balance_sheet)

        print("distribution balance sheet")
        print(distribution_balance_sheet)

        print("transport balance sheet")
        print(transport_balance_sheet)

        print("manufacture balance sheet")
        print(manufacture_balance_sheet)

        print("seller balance sheet")
        print(seller_balance_sheet)

        print("facility balance sheet")
        print(facility_balance_sheet)
