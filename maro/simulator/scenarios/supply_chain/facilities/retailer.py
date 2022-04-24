# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator.scenarios.supply_chain.units import OuterSellerUnit
from maro.simulator.scenarios.supply_chain.units.seller import SellerDemandInterface
from maro.simulator.scenarios.supply_chain.world import World

from .facility import FacilityBase


class RetailerFacility(FacilityBase):
    """Retail facility used to generate order from upstream, and sell products by demand."""
    def __init__(
        self, id: int, name: str, data_model_name: str, data_model_index: int, world: World, config: dict,
    ) -> None:
        super(RetailerFacility, self).__init__(id, name, data_model_name, data_model_index, world, config)


class OuterRetailerFacility(RetailerFacility):
    """Retailer (store) facility that use outer data as seller demand.

    NOTE:
        This require that all product seller is subclass of OuterSellerUnit.
    """

    def __init__(
        self, id: int, name: str, data_model_name: str, data_model_index: int, world: World, config: dict,
    ) -> None:
        super(OuterRetailerFacility, self).__init__(id, name, data_model_name, data_model_index, world, config)

    def initialize(self) -> None:
        super(OuterRetailerFacility, self).initialize()

        assert self.sampler is not None and isinstance(self.sampler, SellerDemandInterface)
        # Go though product to find sellers.
        for product in self.products.values():
            seller = product.seller

            if seller is not None:
                assert issubclass(type(seller), OuterSellerUnit)

                seller.sampler = self.sampler
