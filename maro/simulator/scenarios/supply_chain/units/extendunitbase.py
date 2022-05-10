# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Optional, Union

from .unitbase import BaseUnitInfo, UnitBase

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


@dataclass
class ExtendUnitInfo(BaseUnitInfo):
    product_id: int


class ExtendUnitBase(UnitBase):
    """A base of sku related unit."""

    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict,
    ) -> None:
        super(ExtendUnitBase, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config,
        )

        # Product id (sku id), 0 means invalid.
        self.product_id: int = 0

    def initialize(self) -> None:
        super(ExtendUnitBase, self).initialize()

        if self.data_model is not None:
            self.data_model.set_product_id(self.product_id, self.parent.id)

    def get_unit_info(self) -> ExtendUnitInfo:
        return ExtendUnitInfo(
            **super(ExtendUnitBase, self).get_unit_info().__dict__,
            product_id=self.product_id,
        )
