
from collections import deque, defaultdict

from .base import UnitBase

from typing import Dict


# TODO: original code included the order in raw state, but calculate the price in final state
# so we do not need to put it in the frame, just calculate the total_price per tick/step.
class Order:
    destination = None
    product_id = None
    quantity = None
    vlt = None


class DistributionUnit(UnitBase):
    def __init__(self):
        super(DistributionUnit, self).__init__()

        self.order_queue = deque()

        # used to map from product id to slot index
        self.product_index_mapping: Dict[int, int] = {}

    def initialize(self, configs: dict):
        super(DistributionUnit, self).initialize(configs)

        # create a production index mapping, used to update product information
        for index, product_id in enumerate(self.data.product_list[:]):
            self.product_index_mapping[product_id] = index

    def step(self, tick: int):
        for vehicle in self.facility.transports:
            # if we have vehicle not on the way and there is any pending order
            if len(self.order_queue) > 0 and vehicle.datamodel.location == 0:
                order = self.order_queue.popleft()

                # schedule a job for vehicle
                # TODO: why vlt is determined by order?
                vehicle.schedule(
                    order.destination,
                    order.product_id,
                    order.quantity,
                    order.vlt
                )

        # NOTE: we moved delay_order_penalty from facility to sku, is this ok?
        # update order's delay penalty per tick.
        for order in self.order_queue:
            sku = self.facility.get_sku(order.product_id)
            product_index = self.product_index_mapping[order.product_id]

            self.data.delay_order_penalty[product_index] += sku.delay_order_penalty

    def reset(self):
        super(DistributionUnit, self).reset()
        self.order_queue.clear()

    def get_pending_order(self):
        counter = defaultdict(int)

        for order in self.order_queue:
            counter[order.product_id] += order.quantity

        return counter

    def place_order(self, order):
        if order.quantity > 0:
            sku = self.facility.get_sku_by_id(order.product_id)

            if sku is not None:
                self.order_queue.append(order)

                product_index = self.product_index_mapping[order.product_id]
                order_total_price = sku.price * order.quantity

                self.data.check_in_price[product_index] += order_total_price

                return order_total_price

        return 0
