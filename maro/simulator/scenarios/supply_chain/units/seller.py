
import numpy as np

from .base import UnitBase


class SellerUnit(UnitBase):
    """
    Unit that used to generate product consume demand, and move demand product from current storage.
    """
    def __init__(self):
        super(SellerUnit, self).__init__()

    def initialize(self, configs: dict):
        super(SellerUnit, self).initialize(configs)

    def step(self, tick: int):
        data = self.data

        product_id = data.product_id
        sku = self.facility.sku_information[product_id]
        demand = self.market_demand()

        sold_qty = self.facility.storage.take_available(product_id, demand)

        data.total_sold += sold_qty
        data.sold = sold_qty
        data.demand = demand

        data.balance_sheet_profit = data.unit_price * sold_qty
        data.balance_sheet_loss = -(demand - sold_qty) * data.unit_price * data.backlog_ratio

    def post_step(self, tick: int):
        super(SellerUnit, self).post_step(tick)

        self.data.sold = 0
        self.data.demand = 0

    def reset(self):
        super(SellerUnit, self).reset()

    def market_demand(self):
        return int(np.random.gamma(self.data.sale_gamma))
