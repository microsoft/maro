# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator.scenarios.supply_chain.units import DataFileDemandSampler, OuterSellerUnit
from maro.simulator.scenarios.supply_chain.world import World

from .retailer import RetailerFacility

# Mapping for supported sampler.
sampler_mapping = {
    "data": DataFileDemandSampler,
}


class OuterRetailerFacility(RetailerFacility):
    """Retailer (store) facility that use outer data as seller demand.

    NOTE:
        This require that all product seller is subclass of OuterSellerUnit.
    """

    def __init__(
        self, id: int, name: str, data_model_name: str, data_model_index: int, world: World, config: dict
    ) -> None:
        super(OuterRetailerFacility, self).__init__(id, name, data_model_name, data_model_index, world, config)

    def initialize(self) -> None:
        super(OuterRetailerFacility, self).initialize()

        # What kind of sampler we need?
        sampler_cls = sampler_mapping[self.configs.get("seller_sampler_type", "data")]

        sampler = sampler_cls(self.configs, self.world)

        # Go though product to find sellers.
        for product in self.products.values():
            seller = product.seller

            if seller is not None:
                assert issubclass(type(seller), OuterSellerUnit)

                seller.sampler = sampler
