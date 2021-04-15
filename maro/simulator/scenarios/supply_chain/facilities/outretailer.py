# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .retailer import RetailerFacility
from maro.simulator.scenarios.supply_chain.units import OuterSellerUnit, DataFileDemandSampler


sampler_mapping = {
    "data": DataFileDemandSampler
}


class OutRetailerFacility(RetailerFacility):
    def initialize(self):
        super(OutRetailerFacility, self).initialize()

        # What kind of sampler we need?
        sampler_cls = sampler_mapping[self.configs.get("seller_sampler_type", "data")]

        sampler = sampler_cls(self.configs, self.world)

        # Go though product to find sellers.
        for product in self.products.values():
            seller = product.seller

            if seller is not None:
                assert issubclass(type(seller), OuterSellerUnit)

                seller.sampler = sampler
