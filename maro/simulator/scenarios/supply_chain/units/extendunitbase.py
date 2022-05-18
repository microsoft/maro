# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Optional, Union

from maro.simulator.scenarios.supply_chain.datamodels.extend import ExtendDataModel

from .unitbase import BaseUnitInfo, UnitBase

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase
    from maro.simulator.scenarios.supply_chain.world import World


@dataclass
class ExtendUnitInfo(BaseUnitInfo):
    sku_id: int


class ExtendUnitBase(UnitBase):
    """A base of sku related unit."""

    def __init__(
        self, id: int, data_model_name: Optional[str], data_model_index: Optional[int],
        facility: FacilityBase, parent: Union[FacilityBase, UnitBase], world: World, config: dict,
    ) -> None:
        super(ExtendUnitBase, self).__init__(
            id, data_model_name, data_model_index, facility, parent, world, config,
        )

        # SKU id, 0 means invalid.
        self.sku_id: int = 0

    def initialize(self) -> None:
        super(ExtendUnitBase, self).initialize()

        if self.data_model is not None:
            assert isinstance(self.data_model, ExtendDataModel)
            self.data_model.set_sku_id(self.sku_id, self.parent.id)

    def get_unit_info(self) -> ExtendUnitInfo:
        return ExtendUnitInfo(
            **super(ExtendUnitBase, self).get_unit_info().__dict__,
            sku_id=self.sku_id,
        )
