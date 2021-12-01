# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from itertools import count

from .coordinate import Coordinate

GLOBAL_ORDER_COUNTER = count()


class Order:
    def __init__(self) -> None:
        self.id: str = None
        self.coord: Coordinate = None
        self.privilege = None
        self.open_time = None
        self.close_time = None
        self.is_delivery = None
        self.service_level = None
        self.package_num = None
        self.weight = None
        self.volume = None
        self.creation_time = None
        self.delay_buffer = None
        self.status = None
