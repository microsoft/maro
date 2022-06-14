# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Dict, List

import cvxpy as cp
import numpy as np
import pandas as pd

from maro.rl.policy import RuleBasedPolicy

from .base_policy_data_loader import DataLoaderFromFile, DataLoaderFromHistory


class BaseStockPolicy(RuleBasedPolicy):
    def __init__(self, name: str, policy_parameters: dict) -> None:
        super().__init__(name)

        data_loader_class = eval(policy_parameters["data_loader"])
        assert issubclass(data_loader_class, (DataLoaderFromFile, DataLoaderFromHistory))
        self.data_loader = data_loader_class(policy_parameters)

        self.share_same_stock_level = policy_parameters.get("share_same_stock_level", True)
        self.update_frequency = policy_parameters["update_frequency"]
        self.history_len = policy_parameters["history_len"]
        self.future_len = policy_parameters["future_len"]

        self.product_level_snapshot: Dict[int, Dict[int, int]] = defaultdict(dict)
        self.in_transit_snapshot: Dict[int, Dict[int, int]] = defaultdict(dict)

        self.stock_quantity: Dict[int, Dict[int, int]] = defaultdict(dict)

    def calculate_stock_quantity(
        self,
        input_df: pd.DataFrame,
        product_level: int,
        in_transition_quantity: int,
        vlt: int,
        purchased_before_action: List[int],
    ) -> np.ndarray:
        # time_hrz_len = history_len + 1 + future_len
        time_hrz_len = len(input_df)
        price = np.round(input_df["price"], 1)
        storage_cost = np.round(input_df["storage_cost"], 1)
        order_cost = np.round(input_df["order_cost"], 1)
        demand = np.round(input_df["demand"], 1)

        # Inventory on hand.
        stocks = cp.Variable(time_hrz_len + 1, integer=True)
        # Inventory on the pipeline.
        transits = cp.Variable(time_hrz_len + 1, integer=True)
        sales = cp.Variable(time_hrz_len, integer=True)
        buy = cp.Variable(time_hrz_len + vlt, integer=True)
        # Requested product quantity from upstream.
        buy_in = cp.Variable(time_hrz_len, integer=True)
        # Expected accepted product quantity.
        buy_arv = cp.Variable(time_hrz_len, integer=True)
        target_stock = cp.Variable(time_hrz_len, integer=True)

        profit = cp.Variable(1)

        # Add constraints.
        constraints = [
            # Variable lower bound.
            stocks >= 0,
            transits >= 0,
            sales >= 0,
            buy >= 0,
            # Initial values.
            stocks[0] == product_level,
            transits[0] == in_transition_quantity,
            # Recursion formulas.
            stocks[1 : time_hrz_len + 1] == stocks[0:time_hrz_len] + buy_arv - sales,
            transits[1 : time_hrz_len + 1] == transits[0:time_hrz_len] - buy_arv + buy_in,
            sales <= stocks[0:time_hrz_len],
            sales <= demand,
            buy_in == buy[vlt : time_hrz_len + vlt],
            buy_arv == buy[0:time_hrz_len],
            target_stock == stocks[0:time_hrz_len] + transits[0:time_hrz_len] + buy_in,
            # Objective function.
            profit
            == cp.sum(
                cp.multiply(price, sales) - cp.multiply(order_cost, buy_in) - cp.multiply(storage_cost, stocks[1:]),
            ),
        ]
        # Init the buy before action
        for i in range(vlt):
            constraints.append(buy[i] == purchased_before_action[i])
        obj = cp.Maximize(profit)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.GLPK_MI, verbose=True)
        return target_stock.value

    def _get_action_quantity(self, state: dict) -> int:
        entity_id = state["entity_id"]
        current_tick = state["tick"]
        self.product_level_snapshot[entity_id][current_tick] = state["product_level"]
        self.in_transit_snapshot[entity_id][current_tick] = state["in_transition_quantity"]

        if current_tick % self.update_frequency == 0:
            self.history_start = max(current_tick - self.history_len, 0)
            target_df = self.data_loader.load(state)
            purchased_before_action = [0] * state["cur_vlt"]
            for i in range(min(len(state["history_purchased"]) - 1, state["cur_vlt"])):
                purchased_before_action[-i - 1] = state["history_purchased"][-i - 1]

            self.stock_quantity[entity_id] = self.calculate_stock_quantity(
                target_df,
                self.product_level_snapshot[entity_id][self.history_start],
                self.in_transit_snapshot[entity_id][self.history_start],
                state["cur_vlt"],
                purchased_before_action,
            )

        if self.share_same_stock_level:
            stock_quantity = max(self.stock_quantity[entity_id])
        else:
            stock_quantity = self.stock_quantity[entity_id][current_tick - self.history_start]

        booked_quantity = state["product_level"] + state["in_transition_quantity"]
        quantity = stock_quantity - booked_quantity
        quantity = max(0.0, (1.0 if state["demand_mean"] <= 0.0 else round(quantity / state["demand_mean"], 0)))
        return int(quantity)

    def _rule(self, states: List[dict]) -> List[int]:
        return [self._get_action_quantity(state) for state in states]
