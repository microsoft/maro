# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from itertools import count

from .coordinate import Coordinate

GLOBAL_ORDER_COUNTER = count()

class OrderStatus(Enum):
    # TODO: confirm the order status
    NOT_READY = "order not ready yet"
    IN_PROCESS = "order in process"
    IN_PROCESS_BUT_DELAYED = "order in process but delayed"
    FINISHED = "order finished"
    TERMINATED = "order terminated"


class Order:
    def __init__(self) -> None:
        self.id: str = None
        self.coord: Coordinate = None
        self.privilege = None
        # TODO: align the open time and close time with env tick
        self.open_time = None
        self.close_time = None
        self.is_delivery = None
        self.service_level = None
        self.package_num = None
        self.weight = None
        self.volume = None
        self.creation_time = None
        self.delay_buffer = None
        self._status = OrderStatus.NOT_READY

    @property
    def status(self, tick: int):
        # TODO: update here or in BE?
        if self._status == OrderStatus.NOT_READY and tick >= self.open_time:
            self._status = OrderStatus.IN_PROCESS
        if self._status == OrderStatus.IN_PROCESS and tick > self.close_time:
            self._status = OrderStatus.IN_PROCESS_BUT_DELAYED
        # TODO: logic for terminated?
        return self._status

    @status.setter
    def status(self, var: OrderStatus):
        self._status = var
