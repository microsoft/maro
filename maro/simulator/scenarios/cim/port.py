# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import NodeAttribute, NodeBase, node


@node("ports")
class Port(NodeBase):
    # The capacity of port for stocking containers.
    capacity = NodeAttribute("i")

    # Empty container volume on the port.
    empty = NodeAttribute("i")

    # Laden container volume on the port.
    full = NodeAttribute("i")

    # Empty containers, which are released to the shipper.
    # After loading cargo, laden containers will return to the port for onboarding.
    on_shipper = NodeAttribute("i")

    # Laden containers, which are delivered to the consignee.
    # After discharging cargo, empty containers will return to the port for reuse.
    on_consignee = NodeAttribute("i")

    # Shortage of empty container at current tick.
    # It happens, when the current empty container inventory of port cannot fulfill order requirements.
    shortage = NodeAttribute("i")

    # Accumulated shortage number to the current tick.
    acc_shortage = NodeAttribute("i")

    # Order booking number of a port at the current tick.
    booking = NodeAttribute("i")

    # Accumulated order booking number of a port to the current tick.
    acc_booking = NodeAttribute("i")

    # Fulfilled order number of a port at the current tick.
    fulfillment = NodeAttribute("i")

    # Accumulated fulfilled order number of a port to the current tick.
    acc_fulfillment = NodeAttribute("i")

    # Cost of transferring container, which also covers loading and discharging cost.
    transfer_cost = NodeAttribute("f")

    def __init__(self):
        self._name = None
        self._capacity = None
        self._empty = None

    @property
    def idx(self) -> int:
        """int: Index of this port.
        """
        return self.index

    @property
    def name(self) -> str:
        """str: Name of this port.
        """
        return self._name

    def set_init_state(self, name: str, capacity: int, empty: int):
        """Set initialize state for port, business engine will use these values to reset
        port at the end of each episode (reset).

        Args:
            name (str): Port name.
            capacity (int): Capacity of this port.
            empty (int): Default empty number on this port.
        """
        self._name = name
        self._capacity = capacity
        self._empty = empty

        self.reset()

    def reset(self):
        """Reset port state to initializing.

        Note:
            Since frame reset will reset all the nodes' attributes to 0, we need to
            call set_init_state to store correct initial value.
        """
        self.capacity = self._capacity
        self.empty = self._empty

    def _on_shortage_changed(self, value):
        self._update_fulfilment(value, self.booking)

    def _on_booking_changed(self, value):
        self._update_fulfilment(self.shortage, value)

    def _update_fulfilment(self, shortage: int, booking: int):
        # Update fulfillment.

        self.fulfillment = booking - shortage

    def __str__(self):
        return f"<Port index={self.index}, name={self._name}, capacity={self.capacity}, empty={self.empty}>"
