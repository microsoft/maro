from typing import List

import cvxpy as cp
import numpy as np
import pandas as pd

from .base_policy_data_loader import DataLoaderFromFile, DataLoaderFromHistory
from maro.rl.policy import RuleBasedPolicy


class BaseStockPolicy(RuleBasedPolicy):
    def __init__(self, name: str, policy_para: dict) -> None:
        super().__init__(name)

        data_loader_class = eval(policy_para["data_loader"])
        assert issubclass(data_loader_class, (DataLoaderFromFile, DataLoaderFromHistory))
        self.data_loader = data_loader_class(policy_para)
        self.share_same_stock_level = policy_para.get("share_same_stock_level", True)
        self.update_frequency = policy_para["update_frequency"]
        self.history_len = policy_para["history_len"]
        self.future_len = policy_para["future_len"]

        self.stock_quantity = {}
        self.product_level_snapshot = {}
        self.in_transit_snapshot = {}

    def calculate_stock_quantity(
        self, input_df: pd.DataFrame, product_level: int, in_transition_quantity: int, vlt: int
    ) -> np.ndarray:
        time_hrz_len = len(input_df)
        price = np.round(input_df["price"], 1)
        storage_cost = np.round(input_df["storage_cost"], 1)
        order_cost = np.round(input_df["order_cost"], 1)
        demand = np.round(input_df["demand"], 1)

        stocks = cp.Variable(time_hrz_len + 1, integer=True)
        transits = cp.Variable(time_hrz_len + 1, integer=True)
        sales = cp.Variable(time_hrz_len, integer=True)
        buy = cp.Variable(time_hrz_len * 2, integer=True)
        buy_in = cp.Variable(time_hrz_len, integer=True)
        buy_arv = cp.Variable(time_hrz_len, integer=True)
        inv_pos = cp.Variable(time_hrz_len, integer=True)
        if self.share_same_stock_level:
            target_stock = cp.Variable(1, integer=True)
        else:
            target_stock = cp.Variable(time_hrz_len, integer=True)

        profit = cp.Variable(1)
        buy_in.value = np.ones(time_hrz_len, dtype=np.int)

        # add constraints
        constrs = []
        constrs.extend([
            stocks >= 0, transits >= 0, sales >= 0, buy_in >= 0, buy_arv >= 0, buy >= 0, buy[:time_hrz_len] == 0,
            stocks[0] == product_level,
            transits[0] == in_transition_quantity,
            stocks[1:time_hrz_len + 1] == stocks[0:time_hrz_len] + buy_arv - sales,
            transits[1:time_hrz_len + 1] == transits[0:time_hrz_len] - buy_arv + buy_in,
            sales <= stocks[0:time_hrz_len],
            sales <= demand,
            buy_in == buy[time_hrz_len:],
            inv_pos == stocks[0:time_hrz_len] + transits[0:time_hrz_len],
            buy_arv == buy[time_hrz_len - vlt:2 * time_hrz_len - vlt],
            target_stock == inv_pos + buy_in,
            profit == cp.sum(
                cp.multiply(price, sales)
                - cp.multiply(order_cost, buy_in)
                - cp.multiply(storage_cost, stocks[1:])
            ),
        ])

        obj = cp.Maximize(profit)
        prob = cp.Problem(obj, constrs)
        prob.solve(solver=cp.GLPK_MI, verbose=True)
        return target_stock.value

    def _get_action_quantity(self, state: dict) -> int:
        entity_id = state["entity_id"]

        current_tick = state["tick"]
        history_start = max(current_tick - self.history_len, 0)
        self.product_level_snapshot[current_tick] = state["product_level"]
        self.in_transit_snapshot[current_tick] = state["in_transition_quantity"]
        if current_tick < self.history_len:
            current_index = current_tick
        else:
            current_index = self.history_len + state["tick"] % self.update_frequency
        if current_tick % self.update_frequency == 0:
            target_df = self.data_loader.load(state)
            self.stock_quantity[entity_id] = self.calculate_stock_quantity(
                target_df,
                self.product_level_snapshot[history_start],
                self.in_transit_snapshot[history_start],
                state["cur_vlt"]
            )
        booked_quantity = state["product_level"] + state["in_transition_quantity"]
        if self.share_same_stock_level:
            stock_quantity = self.stock_quantity[entity_id][0]
        else:
            stock_quantity = self.stock_quantity[entity_id][current_index]
        quantity = stock_quantity - booked_quantity
        quantity = max(0.0, (1.0 if state['demand_mean'] <= 0.0 else round(quantity / state['demand_mean'], 0)))
        return int(quantity)

    def _rule(self, states: List[dict]) -> List[int]:
        return [self._get_action_quantity(state) for state in states]
