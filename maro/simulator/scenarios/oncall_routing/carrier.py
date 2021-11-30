# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import NodeAttribute, NodeBase, node

from .coordinate import Coordinate


@node("carriers")
class Carrier(NodeBase):
    latitude = NodeAttribute("f")
    longitude = NodeAttribute("f")

    close_rtb_time = NodeAttribute("i")

    payload_weight = NodeAttribute("f")
    payload_volume = NodeAttribute("f")
    payload_quantity = NodeAttribute("i")

    def __init__(self) -> None:
        self._name = None
        self._route_idx = None
        self._route_name = None

    @property
    def idx(self) -> int:
        return self.index

    @property
    def name(self) -> str:
        return self._name

    @property
    def route_idx(self) -> int:
        return self._route_idx

    @property
    def route_name(self) -> str:
        return self._route_name

    @property
    def coordinate(self) -> Coordinate:
        return Coordinate(lat=self.latitude, lng=self.longitude)

    def set_init_state(
        self, name: str, route_idx: int, route_name: str,
        lat: float, lng: float, close_rtb: int
    ):
        self._name = name
        self._route_idx = route_idx
        self._route_name = route_name

        self._lat = lat
        self._lng = lng
        self._close_rtb = close_rtb

        self.reset()

    def update_coordinate(self, coor: Coordinate):
        self.latitude = coor.lat
        self.longitude = coor.lng

    def reset(self):
        self.latitude = self._lat
        self.longitude = self._lng
        self.close_rtb_time = self._close_rtb
