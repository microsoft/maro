# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .product import ProductUnit
from .unitbase import UnitBase


class VehicleUnit(UnitBase):
    """Unit used to move production from source to destination by order."""

    def __init__(self):
        # Max patient of current vehicle.
        self.max_patient: int = None

        # Current products' destination.
        self.destination: ProductUnit = None

        # Path to destination.
        self.path: list = None

        # Product to load
        self.product_id = 0

        # Steps to arrive destination.
        self.steps = 0

        # Payload on vehicle.
        self.payload = 0

        # Which product unit current product related to.
        self.product: ProductUnit = None

        # Current location in the path.
        self.location = 0

        # Id of destination.
        self.destination_id = 0

        # Velocity.
        self.velocity = 0

    def schedule(self, destination: object, product_id: int, quantity: int, vlt: int):
        """Schedule a job for this vehicle.

        Args:
            destination (FacilityBase): Destination facility.
            product_id (int): What load from storage.
            quantity (int): How many to load.
            vlt (int): Velocity of vehicle.
        """
        # Keep these in states, we will not update it until we reach the destination or cancelled.
        self.data_model.destination = destination.id
        self.data_model.product_id = product_id
        self.data_model.requested_quantity = quantity
        self.data_model.vlt = vlt

        # Cache.
        self.product_id = product_id
        self.destination = destination

        # Keep the patient, reset it after product unloaded.
        self.max_patient = self.data_model.patient

        # Find the path from current entity to target.
        self.path = self.world.find_path(
            self.facility.x,
            self.facility.y,
            destination.x,
            destination.y
        )

        if self.path is None:
            raise Exception(f"Destination {destination} is unreachable")

        # Steps to destination.
        self.steps = len(self.path) // vlt

        # We are waiting for product loading.
        self.location = 0

        self.velocity = vlt

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
        else:
            self.data_model.patient -= 1

            return False

    def try_unload(self):
        """Try unload products into destination's storage."""
        unloaded = self.destination.storage.try_add_products(
            {self.product_id: self.payload},
            all_or_nothing=False
        )

        # Update order if we unloaded any.
        if len(unloaded) > 0:
            unloaded_units = sum(unloaded.values())

            self.destination.products[self.product_id].consumer.on_order_reception(
                self.facility.id,
                self.product_id,
                unloaded_units,
                self.payload
            )

            # Reset the state.
            self.data_model.payload = 0
            self.payload = 0
            self.data_model.patient = self.max_patient
            self.data_model.position[:] = -1

    def initialize(self):
        super(VehicleUnit, self).initialize()

        patient = self.config.get("patient", 100)
        unit_transport_cost = self.config.get("unit_transport_cost", 1)

        self.data_model.initialize(patient=patient, unit_transport_cost=unit_transport_cost)

        self.data_model.position[:] = -1

    def step(self, tick: int):
        # If we have not arrive at destination yet.
        if self.steps > 0:
            # if we still not loaded enough productions yet.
            if self.location == 0 and self.payload == 0:
                # then try to load by requested.
                request_quantity = self.data_model.requested_quantity

                if self.try_load(request_quantity):
                    # NOTE: here we return to simulate loading
                    return
                else:
                    self.data_model.patient -= 1

                    # Failed to load, check the patient.
                    if self.data_model.patient < 0:
                        self.destination.products[self.product_id].consumer.update_open_orders(
                            self.facility.id,
                            self.product_id,
                            -request_quantity
                        )

                        # reset
                        self.steps = 0
                        self.location = 0
                        self.destination = None
                        self.destination_id = 0

                        self._reset_data_model()

            # Moving to destination
            if self.payload > 0:
                # Closer to destination until 0.

                self.location += self.velocity
                self.steps -= 1

                if self.location > len(self.path):
                    self.location = len(self.path) - 1

                self.data_model.position[:] = self.path[self.location]
        else:
            # avoid update under idle state.
            if self.location > 0:
                # try to unload
                if self.payload > 0:
                    self.try_unload()

                # back to source if we unload all
                if self.payload == 0:
                    self.destination = None
                    self.destination_id = 0
                    self.steps = 0
                    self.location = 0
                    self.destination = 0

                    # Reset data model
                    self._reset_data_model()

    def flush_states(self):
        if self.payload > 0:
            self.data_model.payload = self.payload

        if self.steps > 0:
            self.data_model.steps = self.steps

    def reset(self):
        super(VehicleUnit, self).reset()

        self.destination = None
        self.destination_id = 0
        self.max_patient = None
        self.path = None
        self.payload = 0
        self.product_id = 0
        self.steps = 0
        self.location = 0

    def _reset_data_model(self):
        # Reset data model.
        self.data_model.steps = 0
        self.data_model.destination = 0
        self.data_model.product_id = 0
        self.data_model.requested_quantity = 0
        self.data_model.vlt = 0
        self.data_model.payload = 0
        self.data_model.patient = self.max_patient

        self.data_model.position[:] = -1
