

from .base import UnitBase


class ConsumerUnit(UnitBase):
    def __init__(self):
        super(ConsumerUnit, self).__init__()

    def initialize(self, configs: dict):
        super(ConsumerUnit, self).initialize(configs)

    def step(self, tick: int):
        pass

    def reset(self):
        super(ConsumerUnit, self).reset()

    def on_order_reception(self, source_id: int, product_id: int, quantity: int, original_quantity: int):
        pass

    def update_open_orders(self, source_id, product_id, qty_delta):
        pass
