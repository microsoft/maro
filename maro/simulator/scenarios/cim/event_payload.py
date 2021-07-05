# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class VesselStatePayload:
    """Payload object used to hold vessel state changes for event, including VESSEL_ARRIVAL and VESSEL_DEPARTURE.

    Args:
        port_idx (int): Which port the vessel at.
        vessel_idx (int): Which vessel's state changed.
    """
    summary_key = ["port_idx", "vessel_idx"]

    def __init__(self, port_idx: int, vessel_idx: int):

        self.port_idx = port_idx
        self.vessel_idx = vessel_idx

    def __repr__(self):
        return "%s {port_idx: %r, vessel_idx:%r}" % (self.__class__.__name__, self.port_idx, self.vessel_idx)


class VesselDischargePayload:
    """Payload object to hold information about container discharge.

    Args:
        vessel_idx (int): Which vessel will discharge.
        from_port_idx (int): Which port sent the discharged containers.
        port_idx (int): Which port will receive the discharged containers.
        quantity (int): How many containers will be discharged.
    """
    summary_key = ["vessel_idx", "port_idx", "from_port_idx", "quantity"]

    def __init__(self, vessel_idx: int, from_port_idx: int, port_idx: int, quantity: int):
        self.vessel_idx = vessel_idx
        self.from_port_idx = from_port_idx
        self.port_idx = port_idx
        self.quantity = int(quantity)

    def __repr__(self):
        return "%s {port_idx: %r, vessel_idx: %r, quantity: %r, from_port_idx: %r}" % \
            (self.__class__.__name__, self.port_idx, self.vessel_idx, self.quantity, self.from_port_idx)


class LadenReturnPayload:
    """Payload object to hold information about the full return event.

    Args:
        src_port_idx (int): The source port of the laden, i.e, the source port of the corresponding order.
        dest_port_idx (int): Which port the laden are to be sent, i.e, the destination port of the corresponding order.
        quantity (int): How many ladens/containers are returned.
    """
    summary_key = ["src_port_idx", "dest_port_idx", "quantity"]

    def __init__(self, src_port_idx: int, dest_port_idx: int, quantity: int):
        self.src_port_idx = src_port_idx
        self.dest_port_idx = dest_port_idx
        self.quantity = int(quantity)

    def __repr__(self):
        return "%s {src_port_idx: %r, dest_port_idx: %r, quantity:%r}" % \
            (self.__class__.__name__, self.src_port_idx, self.dest_port_idx, self.quantity)


class EmptyReturnPayload:
    """Payload object to hold information about the empty return event.

    Args:
        port_idx (int): Which port the empty containers are returned to.
        quantity (int): How many empty containers are returned.
    """
    summary_key = ["port_idx", "quantity"]

    def __init__(self, port_idx: int, quantity: int):
        self.port_idx = port_idx
        self.quantity = int(quantity)

    def __repr__(self):
        return "%s {port_idx: %r, quantity: %r}" % \
            (self.__class__.__name__, self.port_idx, self.quantity)
