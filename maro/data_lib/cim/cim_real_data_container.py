# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import Dict, List

from .entities import Order, VesselSetting
from .port_buffer_tick_wrapper import PortBufferTickWrapper
from .real_entities import CimRealDataCollection, OrderTuple, RealPortSetting
from .utils import buffer_tick_rand, get_buffer_tick_seed, get_order_num_seed, order_num_rand
from .vessel_future_stops_prediction import VesselFutureStopsPrediction
from .vessel_past_stops_wrapper import VesselPastStopsWrapper
from .vessel_reachable_stops_wrapper import VesselReachableStopsWrapper
from .vessel_sailing_plan_wrapper import VesselSailingPlanWrapper
from .vessel_stop_wrapper import VesselStopsWrapper


class CimRealDataContainer:
    """Data container for cim scenario, used to provide interfaces for business engine,
    and hide the details about data source.

    Args:
        data_collection (CimRealDataCollection): Data collection from source data files.
    """

    def __init__(self, data_collection: CimRealDataCollection):
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

        # orders: the key difference to CimDataCollection
        self._orders: Dict[int, List[OrderTuple]] = self._data_collection.orders

        # keep the seed so we can reproduce the sequence after reset
        self._buffer_tick_seed: int = get_buffer_tick_seed()
        self._order_num_seed: int = get_order_num_seed()

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
    def ports(self) -> List[RealPortSetting]:
        """List[RealPortSetting]: List of port initial settings."""
        return self._data_collection.ports_settings

    @property
    def port_number(self) -> int:
        """int: Number of ports."""
        return len(self._data_collection.ports_settings)

    @property
    def vessels(self) -> List[VesselSetting]:
        """List[VesselSetting]: List of vessel initial settings."""
        return self._data_collection.vessels_settings

    @property
    def vessel_number(self) -> int:
        """int: Number of vessels."""
        return len(self._data_collection.vessels_settings)

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

        orders: List[Order] = [
            Order(order.tick, order.src_port_idx, order.dest_port_idx, order.quantity)
            for order in self._orders[tick]
        ]

        return orders

    def reset(self):
        """Reset data container internal state."""
        self._is_need_reset_seed = True

    def _reset_seed(self):
        """Reset internal seed for generate reproduceable data"""
        buffer_tick_rand.seed(self._buffer_tick_seed)
        order_num_rand.seed(self._order_num_seed)
