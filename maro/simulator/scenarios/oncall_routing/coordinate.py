# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from math import isclose
from typing import NamedTuple


class Coordinate(NamedTuple):
    lat: float
    lng: float

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return isclose(self.lat, other.lat, abs_tol=1e-4) and isclose(self.lng, other.lng, abs_tol=1e-4)
        return False

    def __repr__(self) -> str:
        return f"({self.lat}, {self.lng})"


class CoordinateClipper(object):
    def __init__(self, keep_digit: int) -> None:
        self._keep_digit = keep_digit

    def clip(self, coord: Coordinate) -> Coordinate:
        return Coordinate(
            lat=round(coord.lat, self._keep_digit),
            lng=round(coord.lng, self._keep_digit)
        )
