# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

from maro.simulator.scenarios.supply_chain.order import Order

from .storage import AddStrategy
from .unitbase import BaseUnitInfo, UnitBase

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


class VehicleStatus(Enum):
    Free = 0
    LoadingProducts = 1
    OnTheWayToDestination = 2
    UnloadingProducts = 3


class Vehicle():
    """Unit used to move production from source to destination by order."""

    def __init__(self, facility: FacilityBase, config: dict) -> None:
        self.facility: FacilityBase = facility
        self.config: dict = config

        # Unit cost per quantity.
        self._unit_transport_cost = 0
        # Cost for transportation in current tick.
        self.cost = 0

        # The given product id of this order.
        self.product_id = 0
        # The destination of the product payload.
        self._destination: Optional[FacilityBase] = None

        # Requested product quantity.
        self.requested_quantity = 0
        # The product payload on vehicle.
        self.payload = 0

        # Remaining steps to arrive to the destination.
        self._remaining_steps = 0
        # The steps already traveled from the source facility.
        self._steps = 0
        # Vehicle status
        self._status: VehicleStatus = VehicleStatus.Free

        # Max patient to wait for the try_load() operation, order would be cancelled if patient depleted.
        self._max_patient: int = self.config.get("patient", 100)
        self._remaining_patient = 0

    def schedule(self, destination: FacilityBase, product_id: int, quantity: int, vlt: int) -> None:
        """Schedule a job for this vehicle.

        Args:
            destination (FacilityBase): Destination facility.
            product_id (int): What load from storage.
            quantity (int): How many to load.
            vlt (int): Vendor leading time.
        """
        self.product_id = product_id
        self._destination = destination
        self.requested_quantity = quantity

        # Steps to destination.
        self._remaining_steps = vlt
        self._status = VehicleStatus.LoadingProducts

        dest_consumer = destination.products[product_id].consumer
        if self._remaining_steps < len(dest_consumer.pending_order_daily):
            dest_consumer.pending_order_daily[self._remaining_steps] += quantity

        # We are waiting for product loading.
        self._steps = 0

        self._remaining_patient = self._max_patient

    def try_load(self, quantity: int) -> bool:
        """Try to load specified number of scheduled product.

        Args:
            quantity (int): Number to load.
        """
        if self.facility.storage.try_take_products({self.product_id: quantity}):
            self.payload = quantity

            return True

        return False

    def try_unload(self) -> None:
        """Try unload products into destination's storage."""
        unloaded = self._destination.storage.try_add_products(
            {self.product_id: self.payload},
            add_strategy=AddStrategy.IgnoreUpperBoundAddInOrder,  # TODO: check which strategy to use.
        )

        # Update order if we unloaded any.
        if len(unloaded) > 0:
            assert len(unloaded) == 1
            unloaded_quantity = unloaded[self.product_id]

            self._destination.products[self.product_id].consumer.on_order_reception(
                source_id=self.facility.id,
                product_id=self.product_id,
                received_quantity=unloaded_quantity,
                required_quantity=self.payload,
            )

            self.payload -= unloaded_quantity

    @property
    def status(self) -> VehicleStatus:
        return self._status

    def step(self, tick: int) -> None:
        if self._status == VehicleStatus.LoadingProducts:
            # Try to load by requested.
            if not self.try_load(self.requested_quantity):
                self._remaining_patient -= 1

                # Failed to load, check the patient.
                if self._remaining_patient <= 0:
                    self._destination.products[self.product_id].consumer.update_open_orders(
                        source_id=self.facility.id,
                        product_id=self.product_id,
                        additional_quantity=-self.requested_quantity,
                    )

                    self.reset()
                    # TODO: Add penalty for try-load failure.
                    return
            else:
                self._status = VehicleStatus.OnTheWayToDestination

        if self._status == VehicleStatus.OnTheWayToDestination:
            if self._remaining_steps > 0:
                # Closer to destination until 0.
                self._steps += 1
                self._remaining_steps -= 1

            if self._remaining_steps == 0:
                self._status = VehicleStatus.UnloadingProducts

        if self._status == VehicleStatus.UnloadingProducts:
            # Try to unload.
            if self.payload > 0:
                self.try_unload()  # TODO: to confirm -- the logic is to try unload until all. Add a patient for it?

            # Back to source if we unload all.
            if self.payload == 0:  # TODO: should we simulate the return time cost?
                self.reset()
                return

        self.cost = self.payload * self._unit_transport_cost

    def reset(self) -> None:
        self.cost = 0
        self.product_id = 0
        self._destination = None
        self.requested_quantity = 0
        self.payload = 0
        self._remaining_steps = 0
        self._steps = 0
        self._status = VehicleStatus.Free
        self._remaining_patient = self._max_patient


@dataclass
class DistributionUnitInfo(BaseUnitInfo):
    pass


class DistributionUnit(UnitBase):
    """Unit that used to receive and execute orders from downstream facilities.

    One distribution can accept all kind of sku order.
    """
    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict,
    ) -> None:
        super(DistributionUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config,
        )

        # Vehicle unit dict of this distribution unit. Key: vehicle type; Value: a list of vehicle instances.
        self.vehicles: Dict[str, List[Vehicle]] = defaultdict(list)

        # A dict of pending order queue. Key: vehicle type; Value: pending order queue.
        self._order_queues: Dict[str, deque] = defaultdict(deque)

        # The transportation cost of each product of current tick. Would be set to 0 in ProductUnit.
        # Do not consider the destination here.
        self.transportation_cost = Counter()
        # The delay penalty of each product of current tick. Would be set to 0 in ProductUnit.
        # Do not consider the destination.
        self.delay_order_penalty = Counter()
        # The check-in product quantity in orders of current tick. Would be set to 0 in ProductUnit.
        # Do not consider the destination.
        self.check_in_quantity_in_order = Counter()

        self._unit_delay_order_penalty: Dict[int, float] = {}

        self._is_order_changed: bool = False

    def get_pending_product_quantities(self) -> Dict[int, int]:
        """Count the requested product quantity in pending orders. Only used by BE metrics.

        Returns:
            Dict[int, int]: The key is product id, the value is accumulated requested product quantity.
        """
        counter = defaultdict(int)

        for order_queue in self._order_queues.values():
            for order in order_queue:
                counter[order.product_id] += order.quantity

        for vehicle_list in self.vehicles.values():  # TODO: check whether count these quantity in or not
            for vehicle in vehicle_list:
                if vehicle.status != VehicleStatus.Free:
                    counter[vehicle.product_id] += (vehicle.requested_quantity - vehicle.payload)

        return counter

    def place_order(self, order: Order) -> float:
        """Place an order in the pending order queue, and calculate the corresponding order fee.

        Args:
            order (Order): Order to insert.

        Returns:
            float: The corresponding total order fee, will paid by the consumer.
        """
        if order.quantity > 0:
            sku = self.facility.skus[order.product_id]

            if sku is not None:
                self._is_order_changed = True
                self._order_queues[order.vehicle_type].append(order)
                self.check_in_quantity_in_order[order.product_id] += order.quantity

                order_total_price = sku.price * order.quantity
                return order_total_price

        return 0

    def initialize(self) -> None:
        super(DistributionUnit, self).initialize()

        for vehicle_type, vehicle_config in self.config.items():
            for _ in range(vehicle_config.get("number", 1)):
                vehicle = Vehicle(self.facility, vehicle_config["config"])
                self.vehicles[vehicle_type].append(vehicle)

        for product_id in self.facility.products.keys():
            self._unit_delay_order_penalty[product_id] = self.facility.skus[product_id].unit_delay_order_penalty

    def _step_impl(self, tick: int) -> None:
        # TODO: update vehicle types and distribution step logic
        for vehicle_type, vehicle_list in self.vehicles.items():
            for vehicle in vehicle_list:
                # If we have vehicle not on the way and there is any pending order.
                if len(self._order_queues[vehicle_type]) > 0 and vehicle.requested_quantity == 0:
                    order: Order = self._order_queues[vehicle_type].popleft()

                    # Schedule a job for available vehicle.
                    vehicle.schedule(
                        order.destination,
                        order.product_id,
                        order.quantity,
                        order.vlt,
                    )

                    self._is_order_changed = True

                # Push vehicle.
                vehicle.step(tick)

                self.transportation_cost[vehicle.product_id] += abs(vehicle.cost)

        # Update order's delay penalty per tick.
        for order_queue in self._order_queues.values():
            for order in order_queue:
                self.delay_order_penalty[order.product_id] += self._unit_delay_order_penalty[order.product_id]

    def flush_states(self) -> None:
        super(DistributionUnit, self).flush_states()

        if self._is_order_changed:
            self._is_order_changed = False

            self.data_model.pending_product_quantity = sum(
                order.quantity
                for order_queue in self._order_queues.values()
                for order in order_queue
            )
            self.data_model.pending_order_number = sum(
                len(order_queue)
                for order_queue in self._order_queues.values()
            )

    def reset(self) -> None:
        super(DistributionUnit, self).reset()

        # Reset status in Python side.
        self._order_queues.clear()

        self.transportation_cost.clear()
        self.delay_order_penalty.clear()
        self.check_in_quantity_in_order.clear()

        self._is_order_changed = False

        # Reset vehicles.
        for vehicle_list in self.vehicles.values():
            for vehicle in vehicle_list:
                vehicle.reset()

    def get_unit_info(self) -> DistributionUnitInfo:
        return DistributionUnitInfo(
            **super(DistributionUnit, self).get_unit_info().__dict__,
        )
