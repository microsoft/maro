# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from typing import Optional

from .unitbase import UnitBase
from .storage import AddStrategy

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase


class VehicleUnit(UnitBase):
    """Unit used to move production from source to destination by order."""

    def __init__(self) -> None:
        super().__init__()

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
        # The product yayload on vehicle.
        self.payload = 0

        # Remaining steps to arrive to the destination.
        self._remaining_steps = 0
        # The steps already traveled from the source facility.
        self._steps = 0

        # Max patient to wait for the try_load() operation, order would be cancelled if patient depleted.
        self._max_patient: Optional[int] = None
        self._remaining_patient = 0

    def schedule(self, destination: FacilityBase, product_id: int, quantity: int, vlt: int) -> None:
        """Schedule a job for this vehicle.

        Args:
            destination (FacilityBase): Destination facility.
            product_id (int): What load from storage.
            quantity (int): How many to load.
            vlt (int): Velocity of vehicle.
        """
        self.product_id = product_id
        self._destination = destination
        self.requested_quantity = quantity

        # Steps to destination.
        self._remaining_steps = vlt

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

            # Write to frame, as we do not need to update it per tick.
            self.data_model.payload = quantity

            return True

        return False

    def try_unload(self) -> None:
        """Try unload products into destination's storage."""
        unloaded = self._destination.storage.try_add_products(
            {self.product_id: self.payload},
            add_strategy=AddStrategy.IgnoreUpperBoundAddInOrder  # TODO: check which strategy to use.
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
            self.data_model.payload = self.payload

    def is_enroute(self) -> bool:
        return self._destination is not None

    def initialize(self) -> None:
        super(VehicleUnit, self).initialize()

        self._unit_transport_cost = self.config.get("unit_transport_cost", 1)  # TODO: confirm the default value setting
        self.data_model.initialize(unit_transport_cost=self._unit_transport_cost)

        self._max_patient = self.config.get("patient", 100)  # TODO: confirm the default value setting

    def _step_impl(self, tick: int) -> None:
        # If we have not arrived at destination yet.
        if self._remaining_steps > 0:
            # if we still not loaded enough productions yet.
            if self._steps == 0 and self.payload == 0:
                # then try to load by requested.

                if self.try_load(self.requested_quantity):
                    # NOTE: here we return to simulate loading
                    return
                else:
                    self._remaining_patient -= 1

                    # Failed to load, check the patient.
                    if self._remaining_patient < 0:
                        self._destination.products[self.product_id].consumer.update_open_orders(
                            source_id=self.facility.id,
                            product_id=self.product_id,
                            additional_quantity=-self.requested_quantity,
                        )

                        self._reset_internal_states()
                        self._reset_data_model()

            # Moving to destination
            if self.payload > 0:
                # Closer to destination until 0.
                self._steps += 1
                self._remaining_steps -= 1

        else:
            # Try to unload.
            if self.payload > 0:
                self.try_unload()  # TODO: to confrim -- the logic is to try unload until all

            # Back to source if we unload all.
            if self.payload == 0:  # TODO: should we simulate the return time cost?
                self._reset_internal_states()
                self._reset_data_model()

        self.cost = self.payload * self._unit_transport_cost

    def reset(self) -> None:
        super(VehicleUnit, self).reset()

        # Reset status in Python side.
        self._reset_internal_states()

        self._reset_data_model()

    def _reset_internal_states(self) -> None:
        self.cost = 0
        self.product_id = 0
        self._destination = None
        self.requested_quantity = 0
        self.payload = 0
        self._remaining_steps = 0
        self._steps = 0
        self._remaining_patient = self._max_patient

    def _reset_data_model(self) -> None:
        # Reset data model.
        self.data_model.payload = 0
