# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np


# item used to hold base value and related noise
@dataclass(frozen=True)
class NoisedItem:
    index: int
    base: float
    noise: float


# stop for vessel
@dataclass(frozen=True)
class Stop:
    index: int
    arrival_tick: int
    leave_tick: int
    port_idx: int
    vessel_idx: int


# settings for port
@dataclass(frozen=True)
class PortSetting:
    index: int
    name: str
    capacity: int
    empty: int
    empty_return_buffer: int
    full_return_buffer: int


@dataclass(frozen=True)
class SyntheticPortSetting(PortSetting):
    source_proportion: float
    target_proportions: float


# settings for vessel
@dataclass(frozen=True)
class VesselSetting:
    index: int
    name: str
    capacity: int
    route_name: str
    start_port_name: str
    sailing_speed: float
    sailing_noise: float
    parking_duration: int
    parking_noise: float
    empty: int


# a point in rote definition
@dataclass(frozen=True)
class RoutePoint:
    index: int
    port_name: str
    distance_to_next_port: int


class OrderGenerateMode(Enum):
    """Mode to generate orders from configuration.

    There are 2 modes for now:
    1. fixed: order is generated base on total containers, do not care about available empty container
    2. unfixed: order is generated with configured ratio, and considering available empty containers

    """
    FIXED = "fixed"
    UNFIXED = "unfixed"


# TODO: use object pooling to reduce memory cost
class Order:
    """
    Used to hold order information, this is for order generation.
    """
    summary_key = ["tick", "src_port_idx", "dest_port_idx", "quantity"]

    def __init__(self, tick: int, src_port_idx: int, dest_port_idx: int, quantity: int):
        """
        Create a new instants of order

        Args:
            tick (int): Generated tick of current order.
            src_port_idx (int): Source port of this order.
            dest_port_idx (int): Destination port id of this order.
            quantity (int): Container quantity of this order.
        """
        self.tick = tick
        self.src_port_idx = src_port_idx
        self.quantity = quantity
        self.dest_port_idx = dest_port_idx

    def __repr__(self):
        return "%s {tick: %r, src_port_idx: %r, dest_port_idx: %r, quantity: %r}" % \
            (self.__class__.__name__, self.tick, self.src_port_idx, self.dest_port_idx, self.quantity)


@dataclass(frozen=True)
class CimBaseDataCollection:
    # Port
    port_settings: List[PortSetting]
    port_mapping: Dict[str, int]
    # Vessel
    vessel_settings: List[VesselSetting]
    vessel_mapping: Dict[str, int]
    # Stop
    vessel_stops: List[List[Stop]]
    # Route
    routes: List[List[RoutePoint]]
    route_mapping: Dict[str, int]
    # Vessel Period
    vessel_period_without_noise: List[int]
    # Volume/Container
    container_volume: int
    # Cost Factors
    load_cost_factor: float
    dsch_cost_factor: float
    # Visible Voyage Window
    past_stop_number: int
    future_stop_number: int
    # Time Length of the Data Collection
    max_tick: int
    # Random Seed for Data Generation
    seed: int


# data collection from data generator
@dataclass(frozen=True)
class CimSyntheticDataCollection(CimBaseDataCollection):
    # For Order Generation
    total_containers: int
    order_mode: OrderGenerateMode
    order_proportion: np.ndarray
    # Data Generator Version
    version: str


@dataclass(frozen=True)
class CimRealDataCollection(CimBaseDataCollection):
    # Order Read from Files
    orders: Dict[int, List[Order]]
