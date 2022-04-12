# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator.scenarios.supply_chain.world import World

from .facility import FacilityBase


class SupplierFacility(FacilityBase):
    """Supplier facilities used to produce products with material products."""
    def __init__(
        self, id: int, name: str, data_model_name: str, data_model_index: int, world: World, config: dict,
    ) -> None:
        super(SupplierFacility, self).__init__(id, name, data_model_name, data_model_index, world, config)
