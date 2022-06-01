# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from typing import List, Optional

if typing.TYPE_CHECKING:
    from maro.simulator.scenarios.supply_chain.facilities import FacilityBase


class Order:
    def __init__(
        self,
        src_facility: FacilityBase,
        dest_facility: FacilityBase,
        sku_id: int,
        quantity: int,
        vehicle_type: str,
        creation_tick: int,
        expected_finish_tick: Optional[int],  # It is the expected tick in the moment of taking ConsumerAction.
    ) -> None:
        self.src_facility: FacilityBase = src_facility
        self.dest_facility: FacilityBase = dest_facility
        self.sku_id: int = sku_id
        self.quantity: int = quantity
        self.vehicle_type: str = vehicle_type
        self.creation_tick: int = creation_tick
        self.expected_finish_tick: int = expected_finish_tick

        self.receive_tick_list: List[int] = []
        self.receive_payload_list: List[int] = []
        self.actual_finish_tick: Optional[int] = None
        self._pending_receive_quantity: int = quantity

    @property
    def is_finished(self) -> bool:
        return self._pending_receive_quantity == 0

    def receive(self, tick: int, quantity: int) -> None:
        assert quantity <= self._pending_receive_quantity, (
            f"Only {self._pending_receive_quantity} pending received, but {quantity} got!"
        )

        self.receive_tick_list.append(tick)
        self.receive_payload_list.append(quantity)

        self._pending_receive_quantity -= quantity
        if self._pending_receive_quantity == 0:
            self.actual_finish_tick = tick

        return

    def __repr__(self) -> str:
        return (
            f"Order (created at {self.creation_tick}): "
            f"Ask {self.quantity} products(SKU id: {self.sku_id}) "
            f"from {self.src_facility.name} to {self.dest_facility.name} "
            f"by {self.vehicle_type}"
        )
