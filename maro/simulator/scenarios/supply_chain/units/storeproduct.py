# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from typing import Optional, Union

from .product import ProductUnit
from .unitbase import UnitBase

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


class StoreProductUnit(ProductUnit):
    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict
    ) -> None:
        super(StoreProductUnit, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config
        )

    def get_sale_mean(self) -> float:
        return self.seller.sale_mean()

    def get_sale_std(self) -> float:
        return self.seller.sale_std()

    def get_max_sale_price(self) -> float:
        return self.facility.skus[self.product_id].price
