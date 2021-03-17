# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.simulator.scenarios.supply_chain.actions import ManufactureAction
from .skuunit import SkuUnit


class ManufactureUnit(SkuUnit):
    """Unit that used to produce certain product(sku) with consume specified source skus.

    One manufacture unit per sku.
    """

    # Source material sku and related number per produce cycle.
    bom = None

    # How many production unit each produce cycle.
    output_units_per_lot = None

    # How many unit we will consume each produce cycle.
    input_units_per_lot = 0

    # How many we procedure per current step.
    manufacture_number = 0

    def initialize(self):
        super(ManufactureUnit, self).initialize()

        # TODO: add storage id to data model.
        product_unit_cost = self.config.get("product_unit_cost", 0)

        self.data_model.initialize(
            product_unit_cost=product_unit_cost,
            storage_id=self.facility.storage.id
        )

        # Grab bom of current production.
        sku = self.world.get_sku_by_id(self.product_id)

        self.bom = sku.bom
        self.output_units_per_lot = sku.output_units_per_lot

        if len(self.bom) > 0:
            self.input_units_per_lot = sum(self.bom.values())

    def step(self, tick: int):
        # Try to produce production if we have positive rate.
        if self.action is not None and self.action.production_rate > 0:
            sku_num = len(self.facility.skus)
            unit_num_upper_bound = self.facility.storage.capacity // sku_num

            # Compare with avg storage number.
            current_product_number = self.facility.storage.get_product_number(self.product_id)
            max_number_to_procedure = min(
                unit_num_upper_bound - current_product_number,
                self.action.production_rate * self.output_units_per_lot,
                self.facility.storage.remaining_space
            )

            if max_number_to_procedure > 0:
                space_taken_per_cycle = self.output_units_per_lot - self.input_units_per_lot

                # Consider about the volume, we can produce all if space take per cycle <=1.
                if space_taken_per_cycle > 1:
                    max_number_to_procedure = max_number_to_procedure // space_taken_per_cycle

                source_sku_to_take = {}
                # Do we have enough source material?
                for source_sku_id, source_sku_cost_number in self.bom.items():
                    source_sku_available_number = self.facility.storage.get_product_number(source_sku_id)

                    max_number_to_procedure = min(source_sku_available_number // source_sku_cost_number,
                                                  max_number_to_procedure)

                    if max_number_to_procedure <= 0:
                        break

                    source_sku_to_take[source_sku_id] = max_number_to_procedure * source_sku_cost_number

                if max_number_to_procedure > 0:
                    self.manufacture_number = max_number_to_procedure
                    self.facility.storage.try_take_products(source_sku_to_take)

    def flush_states(self):
        if self.manufacture_number > 0:
            self.data_model.manufacturing_number = self.manufacture_number

    def post_step(self, tick: int):
        if self.manufacture_number > 0:
            self.data_model.manufacturing_number = 0

        if self.action is not None:
            self.data_model.production_rate = 0

        # NOTE: call super at last, since it will clear the action.
        super(ManufactureUnit, self).post_step()

    def set_action(self, action: ManufactureAction):
        super(ManufactureUnit, self).set_action(action)

        self.data_model.production_rate = action.production_rate
