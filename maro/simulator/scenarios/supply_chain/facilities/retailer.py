# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator.scenarios.supply_chain.world import World

from .facility import FacilityBase


class RetailerFacility(FacilityBase):
    """Retail facility used to generate order from upstream, and sell products by demand."""
    def __init__(
        self, id: int, name: str, data_model_name: str, data_model_index: int, world: World, config: dict
    ) -> None:
        super(RetailerFacility, self).__init__(id, name, data_model_name, data_model_index, world, config)
