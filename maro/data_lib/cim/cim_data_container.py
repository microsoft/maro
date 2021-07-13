# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from abc import ABC, abstractmethod
from math import ceil
from typing import Dict, List

from .entities import (
    CimBaseDataCollection, CimRealDataCollection, CimSyntheticDataCollection, NoisedItem, Order, OrderGenerateMode,
    PortSetting, VesselSetting
)
from .port_buffer_tick_wrapper import PortBufferTickWrapper
from .utils import (
    apply_noise, buffer_tick_rand, get_buffer_tick_seed, get_order_num_seed, list_sum_normalize, order_num_rand
)
from .vessel_future_stops_prediction import VesselFutureStopsPrediction
from .vessel_past_stops_wrapper import VesselPastStopsWrapper
from .vessel_reachable_stops_wrapper import VesselReachableStopsWrapper
from .vessel_sailing_plan_wrapper import VesselSailingPlanWrapper
from .vessel_stop_wrapper import VesselStopsWrapper


class CimBaseDataContainer(ABC):
    """Data container for cim scenario, used to provide interfaces for business engine,
    and hide the details about data source, currently we support data from generator, dump files,
    and input following specific schema.

    Example:

        .. code-block:: python

            # Get data from generator.
            data_container = data_from_generator(config_file, max_tick)

            # Get data from dumps folder (which contains several dumped files).
            data_container = data_from_dumps(dump_folder)

            # Get data from input files, the data is organized following specific schema.
            data_container = data_from_files(data_folder)

    Args:
        data_collection (CimBaseDataCollection): Corresponding data collection.
    """
    def __init__(self, data_collection: CimBaseDataCollection):
        self._data_collection = data_collection

        # wrapper for interfaces, to make it easy to use
        self._stops_wrapper = VesselStopsWrapper(self._data_collection)
        self._full_return_buffer_wrapper = PortBufferTickWrapper(
            self._data_collection,
            lambda p: p.full_return_buffer
        )
        self._empty_return_buffer_wrapper = PortBufferTickWrapper(
            self._data_collection,
            lambda p: p.empty_return_buffer
        )
        self._future_stop_prediction = VesselFutureStopsPrediction(self._data_collection)
        self._past_stop_wrapper = VesselPastStopsWrapper(self._data_collection)
        self._vessel_plan_wrapper = VesselSailingPlanWrapper(self._data_collection)
        self._reachable_stops_wrapper = VesselReachableStopsWrapper(self._data_collection)

        # keep the seed so we can reproduce the sequence after reset
        self._buffer_tick_seed: int = get_buffer_tick_seed()

        # flag to tell if we need to reset seed, we need this flag as outside may set the seed after env.reset
        self._is_need_reset_seed = False

    @property
    def past_stop_number(self) -> int:
        """int: Number of past stops to store in snapshot."""
        return self._data_collection.past_stop_number

    @property
    def future_stop_number(self) -> int:
        """int: Number of future stops to predict to store in snapshot."""
        return self._data_collection.future_stop_number

    @property
    def load_cost_factor(self) -> float:
        """float: Factor of the cost for each empty load."""
        return self._data_collection.load_cost_factor

    @property
    def dsch_cost_factor(self) -> float:
        """float: Factor of the cost for each empty discharge."""
        return self._data_collection.dsch_cost_factor

    @property
    def ports(self) -> List[PortSetting]:
        """List[PortSetting]: List of port initial settings."""
        return self._data_collection.port_settings

    @property
    def port_number(self) -> int:
        """int: Number of ports."""
        return len(self._data_collection.port_settings)

    @property
    def vessels(self) -> List[VesselSetting]:
        """List[VesselSetting]: List of vessel initial settings."""
        return self._data_collection.vessel_settings

    @property
    def vessel_number(self) -> int:
        """int: Number of vessels."""
        return len(self._data_collection.vessel_settings)

    @property
    def container_volume(self) -> int:
        """int: Volume of a container.
        """
        # NOTE: we only support 1 type container
        return self._data_collection.container_volume

    @property
    def vessel_stops(self) -> VesselStopsWrapper:
        """Accessor for vessel stops.

        Examples:

            .. code-block:: python

                # Get a stop detail by vessel and location (stop) index.
                stop = data_cntr.vessel_stops[vessel_idx, loc_idx]

                # Get stop list of a vessel.
                stop_list = data_cntr.vessel_stops[vessel_idx]

                # Get all stops, NOTE: slice without parameters.
                stops = data_cntr.vessel_stops[:]
        """
        return self._stops_wrapper

    @property
    def empty_return_buffers(self) -> PortBufferTickWrapper:
        """Accessor to get empty return buffer tick for specified port (with noise).

        Examples:

            .. code-block:: python

                # Get empty return buffer tick of port 0.
                buffer_tick = data_cntr.empty_return_buffers[0]
        """
        return self._empty_return_buffer_wrapper

    @property
    def full_return_buffers(self) -> PortBufferTickWrapper:
        """Accessor to get full return buffer tick for specified port (with noise).

        Examples:

            .. code-block:: python

                # Get full return buffer tick of port 0.
                buffer_tick = data_cnr.full_return_buffers[0]
        """
        return self._full_return_buffer_wrapper

    @property
    def vessel_past_stops(self) -> VesselPastStopsWrapper:
        """Wrapper to get vessel past stops, it will be padding with None if stops number less than configured one.

        Examples:

            .. code-block:: python

                # Get past stops of vessel 0.
                stops = data_cntr.vessel_past_stops[0]
        """
        return self._past_stop_wrapper

    @property
    def vessel_future_stops(self) -> VesselFutureStopsPrediction:
        """Wrapper to get (predict, without noise) vessel future stops, the number of stops is limited by configuration.

        Examples:

            .. code-block:: python

                # Get future stops of vessel 0.
                stops = data_cntr.vessel_future_stops[0]
        """
        return self._future_stop_prediction

    @property
    def vessel_planned_stops(self) -> VesselSailingPlanWrapper:
        """Wrapper to get vessel sailing plan, this method will return a stop list that
        within configured time period (means no same port in list)

        Examples:

            .. code-block:: python

                # Get sailing plan for vessel 0.
                stops = data_cntr.vessel_planned_stops[0]
        """
        return self._vessel_plan_wrapper

    @property
    def reachable_stops(self) -> VesselReachableStopsWrapper:
        """Wrapper to get a list of tuple which contains port index and arrive tick in vessel's route.

        Examples:

            .. code-block:: python

                # Get reachable_stops for vessel 0.
                stop_list = data_cntr.reachable_stops[0]
        """
        return self._reachable_stops_wrapper

    @property
    def vessel_period(self) -> int:
        """Wrapper to get vessel's planed sailing period (without noise to complete a whole route).

        Examples:

            .. code-block:: python

                # Get planed sailing for vessel 0.
                period = data_cntr.vessel_period[0]
        """
        return self._data_collection.vessel_period_without_noise

    @property
    def route_mapping(self) -> Dict[str, int]:
        """Dict[str, int]: Name to index mapping for routes."""
        return self._data_collection.route_mapping

    @property
    def vessel_mapping(self) -> Dict[str, int]:
        """Dict[str, int]: Name to index mapping for vessels."""
        return self._data_collection.vessel_mapping

    @property
    def port_mapping(self) -> Dict[str, int]:
        """Dict[str, int]: Name to index mapping for ports."""
        return self._data_collection.port_mapping

    def reset(self):
        """Reset data container internal state."""
        self._is_need_reset_seed = True

    def _reset_seed(self):
        """Reset internal seed for generate reproduceable data"""
        buffer_tick_rand.seed(self._buffer_tick_seed)

    @abstractmethod
    def get_orders(self, tick: int, total_empty_container: int) -> List[Order]:
        pass


class CimSyntheticDataContainer(CimBaseDataContainer):
    """Data container for synthetic data from generator and dump files for cim scenario.

    Example:

        .. code-block:: python

            # Get data from generator.
            data_container = data_from_generator(config_file, max_tick)

            # Get data from dumps folder (which contains several dumped files).
            data_container = data_from_dumps(dump_folder)

    Args:
        data_collection (CimSyntheticDataCollection): Data collection for synthetic data.
    """

    def __init__(self, data_collection: CimSyntheticDataCollection):
        super().__init__(data_collection)

        # keep the seed so we can reproduce the sequence after reset
        self._order_num_seed: int = get_order_num_seed()

    # TODO: get_events which composed with arrive, departure and order

    def get_orders(self, tick: int, total_empty_container: int) -> List[Order]:
        """Get order list by specified tick.

        Args:
            tick (int): Tick of order.
            total_empty_container (int): Empty container at tick.

        Returns:
            List[Order]: A list of order.
        """

        # reset seed if needed
        if self._is_need_reset_seed:
            self._reset_seed()

            self._is_need_reset_seed = False

        if tick >= self._data_collection.max_tick:
            warnings.warn(f"{tick} out of max tick {self._data_collection.max_tick}")
            return []

        return self._gen_orders(tick, total_empty_container)

    def _reset_seed(self):
        """Reset internal seed for generate reproduceable data"""
        super()._reset_seed()
        order_num_rand.seed(self._order_num_seed)

    def _gen_orders(self, tick: int, total_empty_container: int) -> List[Order]:
        """Generate order for specified tick.

        NOTE:
            Currently we will not dump orders into file even for fixed mode.

        """
        # result
        order_list: List[Order] = []
        order_proportion = self._data_collection.order_proportion
        order_mode = self._data_collection.order_mode
        total_containers = self._data_collection.total_containers

        # max order to gen according to configuration
        orders_to_gen = int(order_proportion[tick])

        # if under unfixed mode, we will consider current empty container as factor
        if order_mode == OrderGenerateMode.UNFIXED:
            delta = total_containers - total_empty_container

            if orders_to_gen <= delta:
                return order_list

            orders_to_gen -= delta

        remaining_orders = orders_to_gen  # used to make sure all the order generated

        # collect and apply noise on the source distribution, then normalized it, to make sure our total number is same
        noised_source_order_dist = []

        # calculate orders distribution for each port as source
        for port_idx in range(self.port_number):
            source_dist: NoisedItem = self.ports[port_idx].source_proportion

            noised_source_order_number = apply_noise(source_dist.base, source_dist.noise, order_num_rand)

            noised_source_order_dist.append(noised_source_order_number)

        # normalize it
        noised_source_order_dist = list_sum_normalize(noised_source_order_dist)

        # generate order for each target port
        for port_idx in range(self.port_number):
            # stop generating if no orders to gen
            if remaining_orders == 0:
                break

            targets_dist: List[NoisedItem] = self.ports[port_idx].target_proportions

            # apply noise and normalize
            noised_targets_dist = list_sum_normalize(
                [apply_noise(target.base, target.noise, order_num_rand) for target in targets_dist])

            # order for current ports
            cur_port_order_num = ceil(orders_to_gen * noised_source_order_dist[port_idx])

            # make sure the total number is correct
            cur_port_order_num = min(cur_port_order_num, remaining_orders)

            # remaining orders for next port
            remaining_orders -= cur_port_order_num

            # dispatch order to targets
            if cur_port_order_num > 0:
                target_remaining_orders = cur_port_order_num

                for i, target in enumerate(targets_dist):
                    # generate and correct order number for each target
                    cur_num = ceil(cur_port_order_num * noised_targets_dist[i])
                    cur_num = min(cur_num, target_remaining_orders)

                    # remaining orders for next target
                    target_remaining_orders -= cur_num

                    # insert into result list
                    if cur_num > 0:
                        order = Order(tick, port_idx, target.index, cur_num)

                        order_list.append(order)

        # TODO: remove later
        assert sum([o.quantity for o in order_list]) == orders_to_gen

        return order_list


class CimRealDataContainer(CimBaseDataContainer):
    """Data container for input data files following a specific data schema as introduced in
    maro\\data_lib\\cim\\README.md.

    Example:

        .. code-block:: python

            # Get data from input files, the data is organized following specific schema.
            data_container = data_from_files(data_folder)
    Args:
        data_collection (CimRealDataCollection): Data collection from source data files.
    """

    def __init__(self, data_collection: CimRealDataCollection):
        super().__init__(data_collection)

        # orders
        self._orders: Dict[int, List[Order]] = self._data_collection.orders

    def get_orders(self, tick: int, total_empty_container: int) -> List[Order]:
        """Get order list by specified tick.

        Args:
            tick (int): Tick of order.
            total_empty_container (int): Empty container at tick.

        Returns:
            List[Order]: A list of order.
        """

        # reset seed if needed
        if self._is_need_reset_seed:
            self._reset_seed()

            self._is_need_reset_seed = False

        if tick >= self._data_collection.max_tick:
            warnings.warn(f"{tick} out of max tick {self._data_collection.max_tick}")
            return []

        if tick not in self._orders:
            return []

        return self._orders[tick]
