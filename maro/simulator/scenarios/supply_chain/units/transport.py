
from .base import UnitBase


class TransportUnit(UnitBase):
    """Unit used to move production from source to destination by order."""
    def __init__(self):
        super().__init__()

        # max patient of current one transport
        self.max_patient = None

        # current products' destination
        self.destination = None

        self.path = None

    def initialize(self, configs: dict):
        super().initialize(configs)

    def reset(self):
        super(TransportUnit, self).reset()

        self.destination = None
        self.max_patient = None
        self.path = None

    def schedule(self, destination, product_id: int, quantity: int, vlt):
        self.data.destination = destination.id
        self.data.product_id = product_id
        self.data.requested_quantity = quantity
        self.data.vlt = vlt

        self.destination = destination
        # keep the patient, reset it after product unloaded.
        self.max_patient = self.data.patient

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
        self.data.steps = len(self.path) // vlt

        # We are waiting for product loading.
        self.data.location = 0

    def try_loading(self, quantity: int):
        if self.facility.storage.try_take_units({self.data.product_id: quantity}):
            self.data.payload = quantity

            return True
        else:
            self.patient -= 1

            return False

    def try_unloading(self):
        unloaded = self.destination.storage.try_add_units(
            {self.data.product_id: self.data.payload},
            all_or_nothing=False
        )

        # update order if we unloaded any
        if len(unloaded) > 0:
            unloaded_units = sum(unloaded.values())

            self.destination.consumers[self.data.product_id].on_order_reception(
                self.facility.id,
                self.data.product_id,
                unloaded_units,
                self.data.payload
            )

            # reset the transport's state
            self.data.payload = 0
            self.data.patient = self.max_patient

    def step(self, tick: int):
        # If we have not arrive at destination yet.
        if self.data.steps > 0:
            # if we still not loaded enough productions yet.
            if self.data.location == 0 and self.data.payload == 0:
                # then try to load by requested.
                if self.try_loading(self.data.requested_quantity):
                    # NOTE: here we return to simulate loading
                    return
                else:
                    self.data.patient -= 1

                    # Failed to load, check the patient.
                    if self.patient < 0:
                        self.destination.consumer.update_open_orders(
                            self.facility.id,
                            self.data.product_id,
                            -self.requested_quantity
                        )

                        # reset
                        self.data.steps = 0
                        self.data.location = 0
                        self.data.destination = 0
                        self.data.position[:] = -1

            # Moving to destination
            if self.data.payload > 0:
                # Closer to destination until 0.

                self.data.location += self.data.vlt
                self.data.steps -= 1

                if self.data.location >= len(self.path):
                    self.data.location = len(self.path) - 1

                self.data.position[:] = self.path[self.data.location]
        else:
            # avoid update under idle state.
            if self.data.location > 0:
                # try to unload
                if self.data.payload > 0:
                    self.try_unloading()

                # back to source if we unload all
                if self.data.payload == 0:
                    self.destination = 0
                    self.data.steps = 0
                    self.data.location = 0
                    self.data.destination = 0
                    self.data.position[:] = -1
