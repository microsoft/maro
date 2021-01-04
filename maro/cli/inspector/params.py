# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class GlobalFileNames:
    ports_sum = "ports_summary.csv"
    vessels_sum = "vessels_summary.csv"
    stations_sum = "stations_summary.csv"
    name_convert = "name_conversion.csv"


class GlobalScenarios(Enum):
    CIM = "container_inventort_management"
    CITI_BIKE = "citi_bike"


class CIMItemOption:
    basic_info = ["name", "frame_index"]
    quick_info = ["All", "Booking Info", "Port Info"]
    acc_info = ["acc_shortage", "acc_booking", "acc_fulfillment"]
    booking_info = ["shortage", "booking", "fulfillment"]
    port_info = ["on_shipper", "on_consignee", "capacity", "full", "empty", "remaining_space"]
    vessel_info = ["capacity", "full", "empty", "remaining_space", "name"]


class CITIBIKEItemOption:
    quick_info = ["All", "Requirement Info", "Station Info"]
    requirement_info = ["trip_requirement", "shortage", "fulfillment"]
    station_info = ["bikes", "capacity", "extra_cost", "failed_return"]
