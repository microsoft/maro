# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from maro.simulator.utils import random

from .coordinate import Coordinate, CoordinateClipper
from .utils import EST_RAND_KEY, geo_distance_meter


@dataclass
class TimePredictionFeature:
    pass


class EstimatedDurationPredictor:
    def __init__(self, coord_clipper: CoordinateClipper, duration_limit: int) -> None:
        self._cache: Dict[Tuple[Coordinate, Coordinate], int] = {}
        self._coord_clipper = coord_clipper
        self._duration_limit = duration_limit

    def predict(
        self,
        tick: int,
        source_coordinate: Coordinate,
        target_coordinate: Coordinate,
        feature: TimePredictionFeature = None
    ) -> int:
        source_coordinate = self._coord_clipper.clip(source_coordinate)
        target_coordinate = self._coord_clipper.clip(target_coordinate)
        min_coord = min(source_coordinate, target_coordinate)
        max_coord = max(source_coordinate, target_coordinate)
        key = (min_coord, max_coord)
        if key not in self._cache:
            if source_coordinate == target_coordinate:
                self._cache[key] = 0
            else:
                distance = geo_distance_meter(source_coordinate, target_coordinate)
                self._cache[key] = int(math.ceil(max(1.0, distance / 200.0)))  # TODO: fake
                self._cache[key] = min(self._cache[key], self._duration_limit)  # TODO
        return self._cache[key]

    def reset(self):
        pass


class ActualDurationSampler:
    def __init__(self, est_duration_predictor: EstimatedDurationPredictor) -> None:
        self._est_duration_predictor = est_duration_predictor

    def sample(
        self,
        tick: int,
        source_coordinate: Coordinate,
        target_coordinate: Coordinate,
        feature: TimePredictionFeature = None
    ) -> int:
        estimated_arrival_time = self._est_duration_predictor.predict(
            tick, source_coordinate, target_coordinate, feature)

        if estimated_arrival_time == 0:
            return estimated_arrival_time
        variance = estimated_arrival_time * 0.1
        noise = random[EST_RAND_KEY].normalvariate(mu=0.0, sigma=variance)
        return int(math.ceil(max(1.0, noise + estimated_arrival_time)))  # TODO: fake

    def reset(self):
        self._est_duration_predictor.reset()
