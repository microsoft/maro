# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .common import Coordinate


# TODO: Rewrite this part in the frame fashion
class Carrier:
    def __init__(self) -> None:
        self.route_number: int = None
        self.coord: Coordinate = None
        self.close_rtb = None
        self.payload_weight = None
        self.payload_volume = None
