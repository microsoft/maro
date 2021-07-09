# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from math import floor

from maro.backends.frame import NodeAttribute, NodeBase, node


def gen_vessel_definition(stop_nums: tuple):
    @node("vessels")
    class Vessel(NodeBase):
        # The capacity of vessel for transferring containers.
        capacity = NodeAttribute("i")

        # Empty container volume on the vessel.
        empty = NodeAttribute("i")

        # Laden container volume on the vessel.
        full = NodeAttribute("i")

        # Remaining space of the vessel.
        remaining_space = NodeAttribute("i")

        # Discharged empty container number for loading laden containers.
        early_discharge = NodeAttribute("i")

        # Is parking or not, 1 means parking, 0 means sailing.
        is_parking = NodeAttribute("i2")

        # The port index the vessel is parking at.
        loc_port_idx = NodeAttribute("i")

        # Which route current vessel belongs to.
        route_idx = NodeAttribute("i")

        # Stop port index in route, it is used to identify where is current vessel.
        # last_loc_idx == next_loc_idx means vessel parking at a port.
        last_loc_idx = NodeAttribute("i")
        next_loc_idx = NodeAttribute("i")

        past_stop_list = NodeAttribute("i", stop_nums[0])
        past_stop_tick_list = NodeAttribute("i", stop_nums[0])
        future_stop_list = NodeAttribute("i", stop_nums[1])
        future_stop_tick_list = NodeAttribute("i", stop_nums[1])

        def __init__(self):
            self._name = None
            self._capacity = None
            self._total_space = None
            self._container_volume = None
            self._route_idx = None
            self._empty = None

        @property
        def name(self) -> str:
            """str: Name of vessel (from config).
            """
            return self._name

        @property
        def idx(self) -> int:
            """int: Index of vessel.
            """
            return self.index

        def set_init_state(self, name: str, container_volume: float, capacity: int, route_idx: int, empty: int):
            """Initialize vessel info that will be used after frame reset.

            Args:
                name (str): Name of vessel.
                container_volume (float): Volume of each container.
                capacity (int): Capacity of this vessel.
                route_idx (int): The index of the route that this vessel belongs to.
                empty (int): Initial empty number of this vessel.
            """
            self._name = name
            self._container_volume = container_volume
            self._total_space = floor(capacity / container_volume)

            self._capacity = capacity
            self._route_idx = route_idx
            self._empty = empty

            self.reset()

        def reset(self):
            """Reset states of vessel."""
            self.capacity = self._capacity
            self.route_idx = self._route_idx
            self.empty = self._empty

        def set_stop_list(self, past_stop_list: list, future_stop_list: list):
            """Set the future stops (configured in config) when the vessel arrive at a port.

            Args:
                past_stop_list (list): List of past stop list tuple.
                future_stop_list (list): List of future stop list tuple.
            """
            # update past and future stop info
            features = []
            if past_stop_list:
                features.append((past_stop_list, self.past_stop_list, self.past_stop_tick_list))
            if future_stop_list:
                features.append((future_stop_list, self.future_stop_list, self.future_stop_tick_list))

            for feature in features:
                for i, stop in enumerate(feature[0]):
                    tick = stop.arrival_tick if stop is not None else -1
                    port_idx = stop.port_idx if stop is not None else -1

                    feature[1][i] = port_idx
                    feature[2][i] = tick

        def _on_empty_changed(self, value):
            self._update_remaining_space()

        def _on_full_changed(self, value):
            self._update_remaining_space()

        def _update_remaining_space(self):
            self.remaining_space = self._total_space - self.full - self.empty

        def __str__(self):
            return f"<Vessel Index={self.index}, capacity={self.capacity}, empty={self.empty}, full={self.full}>"

    return Vessel
