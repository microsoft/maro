# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from geopy.distance import geodesic

from maro.simulator.utils import random

from .common import Coordinate

GLOBAL_RAND_KEY = "global_rand_key"
EST_RAND_KEY = "est_rand_key"
PLAN_RAND_KEY = "plan_rand_key"
ONCALL_RAND_KEY = "oncall_rand_key"

random.create_instance(GLOBAL_RAND_KEY)
random.create_instance(EST_RAND_KEY)
random.create_instance(PLAN_RAND_KEY)
random.create_instance(ONCALL_RAND_KEY)


def geo_distance_meter(source_coordinate: Coordinate, target_coordinate: Coordinate) -> float:
    return geodesic(source_coordinate, target_coordinate).m
