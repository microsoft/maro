# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from dataclasses import dataclass

from maro.simulator.utils import random

from .coordinate import Coordinate
from .utils import EST_RAND_KEY, geo_distance_meter


@dataclass
class TimePredictionFeature:
    pass


class EstimatedDurationPredictor:
    def predict(
        self,
        tick: int,
        source_coordinate: Coordinate,
        target_coordinate: Coordinate,
        feature: TimePredictionFeature = None
    ) -> float:
        distance = geo_distance_meter(source_coordinate, target_coordinate)
        return math.ceil(max(1.0, distance / 200.0))  # TODO: fake


class ActualDurationSampler:
    def sample(
        self,
        tick: int,
        source_coordinate: Coordinate,
        target_coordinate: Coordinate,
        estimated_arrival_time: float
    ) -> float:
        if estimated_arrival_time == 0.0:
            return estimated_arrival_time
        variance = estimated_arrival_time * 0.1
        noise = random[EST_RAND_KEY].normalvariate(mu=0.0, sigma=variance)
        return math.ceil(max(1.0, noise + estimated_arrival_time))  # TODO: fake
