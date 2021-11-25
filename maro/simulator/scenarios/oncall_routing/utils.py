# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from geopy.distance import geodesic

from .common import Coordinate

GLOBAL_RAND_KEY = "global_rand_key"
EST_RAND_KEY = "est_rand_key"


def geo_distance_meter(source_coordinate: Coordinate, target_coordinate: Coordinate) -> float:
    return geodesic(source_coordinate, target_coordinate).m
