
from .base import UnitBase

from .actions import ManufactureAction


class ManufactureUnit(UnitBase):
    """Unit that used to produce certain product(sku) with consume specified source skus.

    One manufacture unit per sku.
    """
    # source material sku and related number per produce cycle
    bom = None

    # how many production unit each produce cycle
    output_units_per_lot = None

    # how many unit we will consume each produce cycle
    input_units_per_lot = 0

    def __init__(self):
        super(ManufactureUnit, self).__init__()

    def initialize(self, configs: dict):
        # add the storage_id
        configs["data"]["storage_id"] = self.facility.storage.id

        super().initialize(configs)

        # grab bom of current production
        sku = self.world.get_sku_by_id(self.data.product_id)
        self.bom = sku.bom
        self.output_units_per_lot = sku.output_units_per_lot

        if len(self.bom) > 0:
            self.input_units_per_lot = sum(self.bom.values())

    def step(self, tick: int):
        # try to produce production if we have positive rate
        data = self.data

        if data.production_rate > 0:
            sku_num = len(self.facility.sku_information)
            unit_num_upper_bound = self.facility.storage.data.capacity // sku_num

            # compare with avg storage number
            current_product_number = self.facility.storage.get_product_number(data.product_id)
            max_number_to_procedure = min(
                unit_num_upper_bound - current_product_number,
                data.production_rate * self.output_units_per_lot,
                self.facility.storage.data.remaining_space
            )

            if max_number_to_procedure > 0:
                space_taken_per_cycle = self.output_units_per_lot - self.input_units_per_lot

                # consider about the volume, we can produce all if space take per cycle <=1
                if space_taken_per_cycle > 1:
                    max_number_to_procedure = max_number_to_procedure // space_taken_per_cycle

                source_sku_to_take = {}
                # do we have enough source material?
                for source_sku_id, source_sku_cost_number in self.bom.items():
                    source_sku_available_number = self.facility.storage.get_product_number(source_sku_id)

                    max_number_to_procedure = min(source_sku_available_number // source_sku_cost_number, max_number_to_procedure)

                    if max_number_to_procedure <= 0:
                        break

                    source_sku_to_take[source_sku_id] = max_number_to_procedure * source_sku_cost_number

                if max_number_to_procedure > 0:
                    data.manufacturing_number += max_number_to_procedure
                    self.facility.storage.try_take_units(source_sku_to_take)

    def post_step(self, tick: int):
        # super(ManufactureUnit, self).post_step(tick)

        # reset the manufacture cost per tick
        self.data.manufacturing_number = 0

    def set_action(self, action: ManufactureAction):
        # we expect production rate number as action
        # production_rate <= 0 will stop manufacturing
        self.data.production_rate = action.production_rate

