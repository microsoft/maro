# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations

import math
import typing
from typing import Optional

from .unitbase import UnitBase

if typing.TYPE_CHECKING:
    from .. import FacilityBase


class VehicleUnit(UnitBase):
    """Unit used to move production from source to destination by order."""

    def __init__(self) -> None:
        super().__init__()
        # Max patient of current vehicle.
        self.max_patient: Optional[int] = None

        # Current products' destination.
        self.destination: Optional[FacilityBase] = None

        # Path to destination.
        self.path: Optional[list] = None

        # Product to load
        self.product_id = 0

        # Steps to arrive destination.
        self.steps = 0

        # Payload on vehicle.
        self.payload = 0

        # Current location in the path.
        self.location = 0

        # Velocity.
        self.velocity = 0
        self.requested_quantity = 0
        self.patient = 0
        self.cost = 0
        self.unit_transport_cost = 0

    def schedule(self, destination: FacilityBase, product_id: int, quantity: int, vlt: int) -> None:
        """Schedule a job for this vehicle.

        Args:
            destination (FacilityBase): Destination facility.
            product_id (int): What load from storage.
            quantity (int): How many to load.
            vlt (int): Velocity of vehicle.
        """
        self.product_id = product_id
        self.destination = destination
        self.requested_quantity = quantity

        # Find the path from current entity to target.
        self.path = self.world.find_path(
            self.facility.x,
            self.facility.y,
            destination.x,
            destination.y,
        )

        if self.path is None:
            raise Exception(f"Destination {destination} is unreachable")

        # Steps to destination.
        self.steps = int(math.ceil(float(len(self.path) - 1) / float(vlt)))
        dest_consumer = destination.products[product_id].consumer
        if self.steps < len(dest_consumer.pending_order_daily):
            dest_consumer.pending_order_daily[self.steps] += quantity

        # We are waiting for product loading.
        self.location = 0

        self.velocity = vlt

        self.patient = self.max_patient

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
        unloaded = self.destination.storage.try_add_products(
            {self.product_id: self.payload},
            all_or_nothing=False,
        )

        # Update order if we unloaded any.
        if len(unloaded) > 0:
            unloaded_units = sum(unloaded.values())

            self.destination.products[self.product_id].consumer.on_order_reception(
                self.facility.id,
                self.product_id,
                unloaded_units,
                self.payload,
            )

            self.payload -= unloaded_units
            self.data_model.payload = self.payload

    def is_enroute(self) -> bool:
        return self.destination is not None

    def initialize(self) -> None:
        super(VehicleUnit, self).initialize()

        patient = self.config.get("patient", 100)
        self.unit_transport_cost = self.config.get("unit_transport_cost", 1)

        self.data_model.initialize(unit_transport_cost=self.unit_transport_cost)

        self.max_patient = patient

    def _step_impl(self, tick: int) -> None:
        # If we have not arrived at destination yet.
        if self.steps > 0:
            # if we still not loaded enough productions yet.
            if self.location == 0 and self.payload == 0:
                # then try to load by requested.

                if self.try_load(self.requested_quantity):
                    # NOTE: here we return to simulate loading
                    return
                else:
                    self.patient -= 1

                    # Failed to load, check the patient.
                    if self.patient < 0:
                        self.destination.products[self.product_id].consumer.update_open_orders(
                            self.facility.id,
                            self.product_id,
                            -self.requested_quantity,
                        )

                        self._reset_internal_states()
                        self._reset_data_model()

            # Moving to destination
            if self.payload > 0:
                # Closer to destination until 0.

                self.location += self.velocity
                self.steps -= 1

                if self.location >= len(self.path):
                    self.location = len(self.path) - 1
        else:
            # Avoid update under idle state.
            # if self.location > 0:
            # Try to unload.///////////////////////////////////////////////////////////////////
            if self.payload > 0:
                self.try_unload()

            # Back to source if we unload all.
            if self.payload == 0:
                self._reset_internal_states()
                self._reset_data_model()

        self.cost = self.payload * self.unit_transport_cost

    def reset(self) -> None:
        super(VehicleUnit, self).reset()

        # Reset status in Python side.
        self._reset_internal_states()

        self._reset_data_model()

    def _reset_internal_states(self) -> None:
        self.destination = None
        self.path = None
        self.product_id = 0
        self.steps = 0
        self.payload = 0
        self.location = 0
        self.velocity = 0
        self.requested_quantity = 0
        self.patient = self.max_patient
        self.cost = 0

    def _reset_data_model(self) -> None:
        # Reset data model.
        self.data_model.payload = 0
