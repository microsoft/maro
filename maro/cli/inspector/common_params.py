import enum as Enum

class CIMItemOption:
    basic_info = ["name", "frame_index"]
    quick_info = ["All", "Booking Info", "Port Info"]
    acc_info = ["acc_shortage", "acc_booking", "acc_fulfillment"]
    booking_info = ["shortage", "booking", "fulfillment"]
    port_info = ["on_shipper", "on_consignee", "capacity", "full", "empty", "remaining_space"]
    vessel_info = ["capacity", "full", "empty", "remaining_space", "name"]


class CITIBIKEOption:
    quick_info = ["All", "Requirement Info", "Station Info"]
    requirement_info = ["trip_requirement", "shortage", "fulfillment"]
    station_info = ["bikes", "capacity", "extra_cost", "failed_return"]


class ScenarioDetail(Enum):
    CIM_Inter = 1
    CIM_Intra = 2
    CITI_BIKE_Summary = 3
    CITI_BIKE_Detail = 4
