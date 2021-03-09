
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
        sku = self.world.get_sku_by_id(self.data.output_product_id)
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

            # one lot per time, until no enough space to hold output, or no enough source material
            # TODO: simplify this part to make it faster
            for _ in range(data.production_rate):
                storage_remaining_space = self.facility.storage.data.remaining_space
                current_product_number = self.facility.storage.get_product_number(data.output_product_id)
                space_taken_per_cycle = self.output_units_per_lot - self.input_units_per_lot

                # if remaining space enough to hold output production
                if storage_remaining_space >= space_taken_per_cycle:
                    # if not reach the storage limitation of current production
                    if current_product_number < unit_num_upper_bound:
                        # if we do not need any material, then just generate the out product.
                        # or if we have enough source materials
                        if len(self.bom) == 0 or self.facility.storage.try_take_units(self.bom):
                            self.facility.storage.try_add_units({data.output_product_id: self.output_units_per_lot})

                            # update manufacturing number in state
                            data.manufacturing_number += 1

        data.balance_sheet_loss = data.manufacturing_number * data.product_unit_cost

    def post_step(self, tick: int):
        super(ManufactureUnit, self).post_step(tick)

        # reset the manufacture cost per tick
        self.data.manufacturing_number = 0

    def set_action(self, action: ManufactureAction):
        # we expect production rate number as action
        # production_rate <= 0 will stop manufacturing
        self.data.production_rate = action.production_rate

