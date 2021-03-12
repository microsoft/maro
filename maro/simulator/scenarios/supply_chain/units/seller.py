
import numpy as np
import random

from .base import UnitBase


class SellerUnit(UnitBase):
    """
    Unit that used to generate product consume demand, and move demand product from current storage.
    """
    def __init__(self):
        super(SellerUnit, self).__init__()

        self.gamma = 0
        self.durations = 0
        self.demand_distribution = []

        # attribute cache
        self.sold = 0
        self.demand = 0
        self.total_sold = 0
        self.product_id = 0

    def initialize(self, configs: dict, durations: int):
        super(SellerUnit, self).initialize(configs, durations)

        self.durations = durations
        self.gamma = self.data.sale_gamma
        self.product_id = self.data.product_id

        for _ in range(durations):
            self.demand_distribution.append(np.random.gamma(self.gamma))

    def step(self, tick: int):
        demand = self.market_demand(tick)

        # what seller does is just count down the product number.
        sold_qty = self.facility.storage.take_available(self.product_id, demand)

        self.total_sold += sold_qty
        self.sold = sold_qty
        self.demand = demand

    def begin_post_step(self, tick: int):
        self.data.sold = self.sold
        self.demand = self.demand
        self.data.total_sold = self.total_sold

    def end_post_step(self, tick: int):
        # super(SellerUnit, self).post_step(tick)
        if self.sold > 0:
            self.data.sold = 0

        if self.demand > 0:
            self.data.demand = 0

    def reset(self):
        super(SellerUnit, self).reset()

        # TODO: regenerate the demand distribution?
        # self.demand_distribution.clear()

        # for _ in range(self.durations):
        #     self.demand_distribution.append(np.random.gamma(self.gamma))

    def market_demand(self, tick: int):
        return int(self.demand_distribution[tick])
