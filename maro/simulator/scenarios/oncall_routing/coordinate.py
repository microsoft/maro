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


def calculate_carrier_coord(
    source_coord: Coordinate,
    target_coord: Coordinate,
    total_time: int,
    passed_time: int
) -> Coordinate:
    if total_time == 0:
        return source_coord
    
    assert 0 <= passed_time <= total_time

    lat_gap = target_coord.lat - source_coord.lat
    lng_gap = target_coord.lng - source_coord.lng

    return Coordinate(
        lat=source_coord.lat + lat_gap * passed_time / total_time,
        lng=source_coord.lng + lng_gap * passed_time / total_time
    )
