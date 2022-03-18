# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .. import ManufactureAction, ManufactureDataModel
from .extendunitbase import ExtendUnitBase


class ManufactureUnit(ExtendUnitBase):
    """Unit that used to produce certain product(sku) with consume specified source skus.

    One manufacture unit per sku.
    """

    def __init__(self) -> None:
        super(ManufactureUnit, self).__init__()

        # Source material sku and related number per produce cycle.
        self._bom: dict = None

        # How many production unit each produce cycle.
        self._output_units_per_lot: int = None

        # How many unit we will consume each produce cycle.
        self._input_units_per_lot: int = 0

        # How many we procedure per current step.
        self._manufacture_number: int = 0

    def initialize(self) -> None:
        super(ManufactureUnit, self).initialize()

        facility_sku_info = self.facility.skus[self.product_id]
        product_unit_cost = facility_sku_info.product_unit_cost

        assert isinstance(self.data_model, ManufactureDataModel)
        self.data_model.initialize(product_unit_cost)

        global_sku_info = self.world.get_sku_by_id(self.product_id)

        self._bom = global_sku_info.bom
        self._output_units_per_lot = global_sku_info.output_units_per_lot

        if len(self._bom) > 0:
            self._input_units_per_lot = sum(self._bom.values())

    def _step_impl(self, tick: int) -> None:
        # Due to the processing in post_step(),
        # self._manufacture_number is set to 0 at the begining of every step.
        # Thus, there is no need to update it with None action or 0 production_rate.

        if self.action is None:
            return

        assert isinstance(self.action, ManufactureAction)

        # Try to produce production if we have positive rate.
        if self.action.production_rate > 0:
            sku_num = len(self.facility.skus)
            unit_num_upper_bound = self.facility.storage.capacity // sku_num

            # Compare with avg storage number.
            current_product_quantity = self.facility.storage.get_product_quantity(self.product_id)
            max_number_to_procedure = min(
                unit_num_upper_bound - current_product_quantity,
                self.action.production_rate * self._output_units_per_lot,
                self.facility.storage.remaining_space,
            )

            if max_number_to_procedure > 0:
                space_taken_per_cycle = self._output_units_per_lot - self._input_units_per_lot

                # Consider about the volume, we can produce all if space take per cycle <=1.
                if space_taken_per_cycle > 1:
                    max_number_to_procedure = max_number_to_procedure // space_taken_per_cycle

                # Do we have enough source material?
                for source_sku_id, source_sku_cost_number in self._bom.items():
                    source_sku_available_number = self.facility.storage.get_product_quantity(source_sku_id)

                    max_number_to_procedure = min(
                        source_sku_available_number // source_sku_cost_number,
                        max_number_to_procedure,
                    )

                    if max_number_to_procedure <= 0:
                        break

                if max_number_to_procedure > 0:
                    source_sku_to_take = {}
                    for source_sku_id, source_sku_cost_number in self._bom.items():
                        source_sku_to_take[source_sku_id] = max_number_to_procedure * source_sku_cost_number

                    self._manufacture_number = max_number_to_procedure
                    self.facility.storage.try_take_products(source_sku_to_take)
                    self.facility.storage.try_add_products({self.product_id: self._manufacture_number})

    def flush_states(self) -> None:
        if self._manufacture_number > 0:
            self.data_model.manufacturing_number = self._manufacture_number

    def post_step(self, tick: int) -> None:
        if self._manufacture_number > 0:
            self.data_model.manufacturing_number = 0
            self._manufacture_number = 0

        # NOTE: call super at last, since it will clear the action.
        super(ManufactureUnit, self).post_step(tick)

    def reset(self) -> None:
        super(ManufactureUnit, self).reset()

        # Reset status in Python side.
        self._manufacture_number = 0
