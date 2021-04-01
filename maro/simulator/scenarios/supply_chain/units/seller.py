# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import numpy as np

from .skuunit import SkuUnit


class SellerUnit(SkuUnit):
    """
    Unit that used to generate product consume demand, and move demand product from current storage.
    """

    def __init__(self):
        super(SellerUnit, self).__init__()

        self.gamma = 0
        self.durations = 0
        self.demand_distribution = []

        # Attribute cache.
        self.sold = 0
        self.demand = 0
        self.total_sold = 0
        self.total_demand = 0
        self.price = 0

        self.sale_hist = []
        self.backlog_ratio = 0

    def market_demand(self, tick: int) -> int:
        """Generate market demand for current tick.

        Args:
            tick (int): Current simulator tick.

        Returns:
            int: Demand number.
        """
        return self.demand_distribution[tick]

    def initialize(self):
        super(SellerUnit, self).initialize()

        sku = self.facility.skus[self.product_id]

        self.price = sku.price
        self.gamma = sku.sale_gamma
        self.backlog_ratio = sku.backlog_ratio
        self.durations = self.world.durations

        # Generate demand distribution of this episode.
        for _ in range(self.durations):
            self.demand_distribution.append(int(np.random.gamma(self.gamma)))

        self.sale_hist = [self.gamma] * self.config["sale_hist_len"]

    def step(self, tick: int):
        demand = self.market_demand(tick)

        # What seller does is just count down the product number.
        sold_qty = self.facility.storage.take_available(self.product_id, demand)

        self.total_sold += sold_qty
        self.sold = sold_qty
        self.demand = demand
        self.total_demand += demand

        self.sale_hist.append(demand)
        self.sale_hist = self.sale_hist[1:]

        self.step_balance_sheet.profit = sold_qty * self.price
        self.step_balance_sheet.loss = -demand * self.price * self.backlog_ratio
        self.step_reward = self.step_balance_sheet.total()

    def flush_states(self):
        if self.sold > 0:
            self.data_model.sold = self.sold
            self.data_model.total_sold = self.total_sold

        if self.demand > 0:
            self.data_model.demand = self.demand
            self.data_model.total_demand = self.total_demand

    def post_step(self, tick: int):
        super(SellerUnit, self).post_step(tick)

        if self.sold > 0:
            self.data_model.sold = 0
            self.sold = 0

        if self.demand > 0:
            self.data_model.demand = 0
            self.demand = 0

    def reset(self):
        super(SellerUnit, self).reset()

        # TODO: regenerate the demand distribution?
        # self.demand_distribution.clear()

        # for _ in range(self.durations):
        #     self.demand_distribution.append(np.random.gamma(self.gamma))

    def sale_mean(self):
        return np.mean(self.sale_hist)

    def sale_std(self):
        return np.std(self.sale_hist)
