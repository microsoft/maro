# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from typing import Optional

from .unitbase import UnitBase

if typing.TYPE_CHECKING:
    from .. import FacilityBase


class VehicleUnit(UnitBase):
    """Unit used to move production from source to destination by order."""

    def __init__(self) -> None:
        super().__init__()
        # Max patient to wait for the try_load() operation.
        self._max_patient: Optional[int] = None

        self._unit_transport_cost = 0

        # Current products' destination.
        self._destination: Optional[FacilityBase] = None

        # Path to destination.
        self._path: Optional[list] = None

        # Product to load
        self.product_id = 0

        # Steps to arrive destination.
        self._remaining_steps = 0

        # Payload on vehicle.
        self.payload = 0

        # Current location in the path.
        self._location = 0

        # Velocity.
        self._velocity = 0
        self.requested_quantity = 0
        self._remaining_patient = 0
        self.cost = 0

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

        # Find the path from current entity to target.
        self._path = self.world.find_path(
            self.facility.x,
            self.facility.y,
            destination.x,
            destination.y,
        )

        if self._path is None:
            raise Exception(f"Destination {destination} is unreachable")

        # Steps to destination.
        self._remaining_steps = vlt
        self._velocity = 1

        dest_consumer = destination.products[product_id].consumer
        if self._remaining_steps < len(dest_consumer.pending_order_daily):
            dest_consumer.pending_order_daily[self._remaining_steps] += quantity

        # We are waiting for product loading.
        self._location = 0

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
            all_or_nothing=False,
        )

        # Update order if we unloaded any.
        if len(unloaded) > 0:
            unloaded_units = sum(unloaded.values())

            self._destination.products[self.product_id].consumer.on_order_reception(
                self.facility.id,
                self.product_id,
                unloaded_units,
                self.payload,
            )

            self.payload -= unloaded_units
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
            if self._location == 0 and self.payload == 0:
                # then try to load by requested.

                if self.try_load(self.requested_quantity):
                    # NOTE: here we return to simulate loading
                    return
                else:
                    self._remaining_patient -= 1

                    # Failed to load, check the patient.
                    if self._remaining_patient < 0:
                        self._destination.products[self.product_id].consumer.update_open_orders(
                            self.facility.id,
                            self.product_id,
                            -self.requested_quantity,
                        )

                        self._reset_internal_states()
                        self._reset_data_model()

            # Moving to destination
            if self.payload > 0:
                # Closer to destination until 0.
                self._location += self._velocity
                self._remaining_steps -= 1

                if self._location >= len(self._path):  # TODO: will we look into the path with location as the index?
                    self._location = len(self._path) - 1
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
        self._destination = None
        self._path = None
        self.product_id = 0
        self._remaining_steps = 0
        self.payload = 0
        self._location = 0
        self._velocity = 0
        self.requested_quantity = 0
        self._remaining_patient = self._max_patient
        self.cost = 0

    def _reset_data_model(self) -> None:
        # Reset data model.
        self.data_model.payload = 0
