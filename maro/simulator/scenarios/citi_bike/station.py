# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.frame import NodeAttribute, NodeBase, node


@node("stations")
class Station(NodeBase):
    """Station node definition in frame."""

    bikes = NodeAttribute("i")

    # statistics features
    shortage = NodeAttribute("i")
    trip_requirement = NodeAttribute("i")
    fulfillment = NodeAttribute("i")

    capacity = NodeAttribute("i")
    id = NodeAttribute("i")

    # additional features
    weekday = NodeAttribute("i2")

    # avg temp
    temperature = NodeAttribute("i2")

    # 0: sunny, 1: rainy, 2: snowy， 3: sleet
    weather = NodeAttribute("i2")

    # 0: holiday, 1: not holiday
    holiday = NodeAttribute("i2")

    extra_cost = NodeAttribute("i")
    transfer_cost = NodeAttribute("i")
    failed_return = NodeAttribute("i")

    # min bikes between a frame
    min_bikes = NodeAttribute("i")

    def __init__(self):
        # internal use for reset
        self._init_capacity = 0
        # internal use for reset
        self._init_bikes = 0
        # original id in data file
        self._id = 0

    def set_init_state(self, bikes: int, capacity: int, id: int):
        """Set initialize state, that will be used after frame reset.

        Args:
            bikes (int): Total bikes on this station.
            capacity (int): How many bikes this station can hold.
            id (int): Id of this station.
        """
        self._init_bikes = bikes
        self._init_capacity = capacity
        self._id = id

        self.reset()

    def reset(self):
        """Reset to default value."""
        # when we reset frame, all the value will be set to 0, so we need these lines
        self.capacity = self._init_capacity
        self.bikes = self._init_bikes
        self.min_bikes = self._init_bikes
        self.id = self._id

    def _on_bikes_changed(self, value: int):
        """Update min bikes after bikes changed"""
        cur_min_bikes = self.min_bikes

        self.min_bikes = min(value, cur_min_bikes)


def gen_matrices_node_definition(station_num: int):
    """Function to generate adj node definition, due to we need the numbers at runtime.

    Args:
        station_num (int): Total stations of current simulation.

    Returns:
        type: Node definition class for matrices.
    """

    @node("matrices")
    class Matrices(NodeBase):
        trips_adj = NodeAttribute("i", station_num * station_num)

        def reset(self):
            pass

    return Matrices
