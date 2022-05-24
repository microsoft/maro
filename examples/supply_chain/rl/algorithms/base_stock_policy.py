
import numpy as np
import cvxpy as cp
from .rule_based import RuleBasedPolicy
from typing import List

class BaseStockPolicy(RuleBasedPolicy):
    def __init__(self, name: str, policy_para) -> None:
        super().__init__(name)

        self.update_frequency = policy_para["update_frequency"]
        self.data_loader = policy_para["data_loader"]
        self.start_index = 0
        self.step = {}
        self.stock_quantity = {}

    def calculate_stock_quantity(self, input_df, state):
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
        target_stock = cp.Variable(time_hrz_len, integer=True)
        profit = cp.Variable(1)
        buy_in.value = np.ones(time_hrz_len, dtype = np.int)
        
        #add constraints
        constrs=[]
        constrs.extend([
            stocks >= 0, transits >= 0, sales >= 0, buy_in>=0, buy_arv>=0, buy>=0, buy[:time_hrz_len]==0,
            stocks[0] == state["product_level"],
            transits[0] == state["in_transition_quantity"],
            stocks[1:time_hrz_len+1] == stocks[0:time_hrz_len] + buy_arv - sales,
            transits[1:time_hrz_len+1] == transits[0:time_hrz_len] - buy_arv + buy_in,
            sales <= stocks[0:time_hrz_len],
            sales <= demand,
            buy_in == buy[time_hrz_len:],
            inv_pos == stocks[0:time_hrz_len] + transits[0:time_hrz_len],
            buy_arv == buy[time_hrz_len - state["cur_vlt"] : 2 * time_hrz_len - state["cur_vlt"]],
            target_stock == inv_pos + buy_in,
            profit == cp.sum(cp.multiply(price, sales) - cp.multiply(order_cost, buy_in) - cp.multiply(storage_cost, stocks[1:])),
        ])
        
        obj = cp.Maximize(profit)
        prob = cp.Problem(obj, constrs)
        prob.solve(solver = cp.GLPK_MI, verbose = True)
        return target_stock.value

    def _get_action_quantity(self, state: dict) -> int:
        entity_id = state["entity_id"]
        if entity_id not in self.step or self.step[entity_id] == self.update_frequency:
            self.step[entity_id] = 0
            target_df, today_index = self.data_loader.load(state)
            self.stock_quantity[entity_id] = self.calculate_stock_quantity(target_df, state)[today_index:]
        booked_quantity = state["product_level"] + state["in_transition_quantity"]# - state["to_distribute_quantity"]
        quantity = self.stock_quantity[entity_id][self.step[entity_id]] - booked_quantity
        quantity = max(0.0, (1.0 if state['demand_mean'] <= 0.0 else round(quantity / state['demand_mean'], 0)))
        self.step[entity_id] += 1
        return int(quantity)
    
    def _rule(self, states: List[dict]) -> List[int]:
        return [self._get_action_quantity(state) for state in states]
