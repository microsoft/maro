# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.simulator.graph import Graph, GraphAttributeType

AT_DYNAMIC = GraphAttributeType.DYNAMIC_NODE

class Vessel:
    """
    Wrapper that present a vessel in ECR problem and hide the detail of graph accessing
    """

    def __init__(self, graph: Graph, idx: int, name: str):
        """
        Create a new instance of vessel

        Args:
            graph (Graph): graph that the vessel belongs to
            idx (int): index of this vessel
            name (str): name of this vessel
        """
        self._graph = graph
        self._idx = idx
        self._name = name

    def reset(self):
        """
        Reset current vessel state
        """
        pass

    @property
    def idx(self) -> int:
        """
        Index of vessel
        """
        return self._idx

    @property
    def name(self) -> str:
        """
        Name of vessel (from config)
        """
        return self._name

    @property
    def capacity(self) -> float:
        """
        Capacity of vessel, when the contains on board reach the capacity, the vessel cannot load any container
        """
        return self._graph.get_attribute(AT_DYNAMIC, self._idx, "capacity", 0)

    @capacity.setter
    def capacity(self, value: float):
        self._graph.set_attribute(AT_DYNAMIC, self._idx, "capacity", 0, value)

    @property
    def empty(self) -> int:
        """
        Number of empty containers on board
        """
        return self._graph.get_attribute(AT_DYNAMIC, self._idx, "empty", 0)

    @empty.setter
    def empty(self, value: int):
        self._graph.set_attribute(AT_DYNAMIC, self._idx, "empty", 0, value)

    @property
    def full(self) -> int:
        """
        Number of full on board
        """
        return self._graph.get_attribute(AT_DYNAMIC, self._idx, "full", 0)

    @full.setter
    def full(self, value: int):
        self._graph.set_attribute(AT_DYNAMIC, self._idx, "full", 0, value)

    @property
    def early_discharge(self) -> int:
        """
        Number of full on board
        """
        return self._graph.get_attribute(AT_DYNAMIC, self._idx, "early_discharge", 0)

    @early_discharge.setter
    def early_discharge(self, value: int):
        self._graph.set_attribute(AT_DYNAMIC, self._idx, "early_discharge", 0, value)

    @property
    def last_loc_idx(self) -> int:
        """
        Last location index in loop
        """
        return self._graph.get_attribute(AT_DYNAMIC, self._idx, "last_loc_idx", 0)

    @last_loc_idx.setter
    def last_loc_idx(self, value: int):
        self._graph.set_attribute(AT_DYNAMIC, self._idx, "last_loc_idx", 0, value)

    @property
    def next_loc_idx(self) -> int:
        """
        Next location index in loop
        """
        return self._graph.get_attribute(AT_DYNAMIC, self._idx, "next_loc_idx", 0)

    @next_loc_idx.setter
    def next_loc_idx(self, value: int):
        self._graph.set_attribute(AT_DYNAMIC, self._idx, "next_loc_idx", 0, value)

    @property
    def route_idx(self) -> int:
        """
        Index of route this vessel belongs to
        """
        return self._graph.get_attribute(AT_DYNAMIC, self._idx, "route_idx", 0)

    @route_idx.setter
    def route_idx(self, value: int):
        self._graph.set_attribute(
            AT_DYNAMIC, self._idx, "route_idx", 0, value)

    @property
    def remaining_space(self):
        return self._graph.get_attribute(AT_DYNAMIC, self._idx, "remaining_space", 0)

    @remaining_space.setter
    def remaining_space(self, value: float):
        self._graph.set_attribute(AT_DYNAMIC, self._idx, "remaining_space", 0, value)

    def set_stop_list(self, stop_list: tuple):
        """
        Set the future stops (configured in config) when the vessel arrive at a port

        Args:
            stop_list (tuple): list of past and future stop list tuple
        """
        features = [(stop_list[0], "past_stop_list", "past_stop_tick_list"),
                    (stop_list[1], "future_stop_list", "future_stop_tick_list")]

        for feature in features:
            for i, stop in enumerate(feature[0]):
                tick = stop.arrive_tick if stop is not None else -1
                port_idx = stop.port_idx if stop is not None else -1

                self._graph.set_attribute(
                    AT_DYNAMIC, self._idx, feature[1], i, port_idx)
                self._graph.set_attribute(
                    AT_DYNAMIC, self._idx, feature[2], i, tick)
