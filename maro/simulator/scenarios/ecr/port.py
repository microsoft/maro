# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.simulator.graph import Graph, ResourceNodeType


class Port:
    """
    Present a port in ECR problem, and hide detail of graph accessing
    """

    def __init__(self, graph: Graph, idx: int, name: str):
        """
        Create a new instance of port

        Args:
            graph (Graph): graph this port belongs to
            idx (int): index of this port
            name (str): name of this port
        """
        self._graph = graph
        self._idx = idx
        self._name = name

    def reset(self):
        """
        reset current port
        """
        pass

    @property
    def idx(self) -> int:
        """
        Index of this prot
        """
        return self._idx

    @property
    def name(self) -> str:
        """
        Name of this port
        """
        return self._name

    @property
    def empty(self) -> int:
        """
        Number of empty containers on board
        """
        return self._graph.get_attribute(ResourceNodeType.STATIC, self._idx, "empty", 0)

    @empty.setter
    def empty(self, value: int):
        self._graph.set_attribute(ResourceNodeType.STATIC, self._idx, "empty", 0, value)

    @property
    def full(self) -> int:
        """
        Number of full containers on board
        """
        return self._graph.get_attribute(ResourceNodeType.STATIC, self._idx, "full", 0)

    @full.setter
    def full(self, value: int):
        self._graph.set_attribute(ResourceNodeType.STATIC, self._idx, "full", 0, value)

    @property
    def on_shipper(self) -> int:
        """
        Number of empty containers that will become full, and need time to return the port

        """
        return self._graph.get_attribute(ResourceNodeType.STATIC, self._idx, "on_shipper", 0)

    @on_shipper.setter
    def on_shipper(self, value: int):
        self._graph.set_attribute(ResourceNodeType.STATIC, self._idx, "on_shipper", 0, value)

    @property
    def on_consignee(self) -> int:
        """
        Number of full containers that discharged at this port, and need time to become empty container
        """
        return self._graph.get_attribute(ResourceNodeType.STATIC, self._idx, "on_consignee", 0)

    @on_consignee.setter
    def on_consignee(self, value: int):
        self._graph.set_attribute(ResourceNodeType.STATIC, self._idx, "on_consignee", 0, value)

    @property
    def shortage(self) -> int:
        """
        Shortage of containers on this port at current tick
        """
        return self._graph.get_attribute(ResourceNodeType.STATIC, self._idx, "shortage", 0)

    @shortage.setter
    def shortage(self, value: int):
        self._graph.set_attribute(ResourceNodeType.STATIC, self._idx, "shortage", 0, value)

        self._update_fulfilment(value, self.booking)

    @property
    def acc_shortage(self) -> int:
        """
        accumulative shortage to current tick
        """
        return self._graph.get_attribute(ResourceNodeType.STATIC, self._idx, "acc_shortage", 0)

    @acc_shortage.setter
    def acc_shortage(self, value: int):
        self._graph.set_attribute(ResourceNodeType.STATIC, self._idx, "acc_shortage", 0, value)

    @property
    def capacity(self) -> float:
        """
        Capacity of this port
        """
        return self._graph.get_attribute(ResourceNodeType.STATIC, self._idx, "capacity", 0)

    @capacity.setter
    def capacity(self, value: float):
        self._graph.set_attribute(ResourceNodeType.STATIC, self._idx, "capacity", 0, value)

    @property
    def booking(self) -> int:
        """
        Booking number of this port at current tick
        """
        return self._graph.get_attribute(ResourceNodeType.STATIC, self._idx, "booking", 0)

    @booking.setter
    def booking(self, value: int):
        self._graph.set_attribute(ResourceNodeType.STATIC, self._idx, "booking", 0, value)

        self._update_fulfilment(self.shortage, value)

    @property
    def acc_booking(self) -> int:
        """
        Accumulative booking number of this port
        """
        return self._graph.get_attribute(ResourceNodeType.STATIC, self._idx, "acc_booking", 0)

    @acc_booking.setter
    def acc_booking(self, value: int):
        self._graph.set_attribute(ResourceNodeType.STATIC, self._idx, "acc_booking", 0, value)

    @property
    def fulfillment(self) -> int:
        return self._graph.get_attribute(ResourceNodeType.STATIC, self._idx, "fulfillment", 0)

    @fulfillment.setter
    def fulfillment(self, value: int):
        """
        Fulfillment of current tick
        """
        self._graph.set_attribute(ResourceNodeType.STATIC, self._idx, "fulfillment", 0, value)

    @property
    def acc_fulfillment(self) -> int:
        return self._graph.get_attribute(ResourceNodeType.STATIC, self._idx, "acc_fulfillment", 0)

    @acc_fulfillment.setter
    def acc_fulfillment(self, value: int):
        self._graph.set_attribute(ResourceNodeType.STATIC, self._idx, "acc_fulfillment", 0, value)

    def _update_fulfilment(self, shortage: int, booking: int):
        # update fulfillment

        self.fulfillment = booking - shortage
