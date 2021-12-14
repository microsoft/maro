# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from maro.simulator.utils import random

from .coordinate import Coordinate
from .utils import EST_RAND_KEY, geo_distance_meter


@dataclass
class TimePredictionFeature:
    pass


class EstimatedDurationPredictor:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[Coordinate, Coordinate], int] = {}

    def predict(
        self,
        tick: int,
        source_coordinate: Coordinate,
        target_coordinate: Coordinate,
        feature: TimePredictionFeature = None
    ) -> int:
        min_coord = min(source_coordinate, target_coordinate)
        max_coord = max(source_coordinate, target_coordinate)
        key = (min_coord, max_coord)
        if key not in self._cache:
            distance = geo_distance_meter(source_coordinate, target_coordinate)
            self._cache[key] = int(math.ceil(max(1.0, distance / 200.0)))  # TODO: fake
        return self._cache[key]

    def reset(self):
        pass


class ActualDurationSampler:
    def sample(
        self,
        tick: int,
        source_coordinate: Coordinate,
        target_coordinate: Coordinate,
        estimated_arrival_time: int
    ) -> int:
        if estimated_arrival_time == 0.0:
            return estimated_arrival_time
        variance = estimated_arrival_time * 0.1
        noise = random[EST_RAND_KEY].normalvariate(mu=0.0, sigma=variance)
        return int(math.ceil(max(1.0, noise + estimated_arrival_time)))  # TODO: fake

    def reset(self):
        pass
