# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class SupplyChainAction:
    def __init__(self, id: int) -> None:
        self.id = id


class ConsumerAction(SupplyChainAction):
    def __init__(
        self, id: int, product_id: int, source_id: int, quantity: int, vlt: int, reward_discount: float,
    ) -> None:
        super(ConsumerAction, self).__init__(id=id)
        self.product_id = product_id
        self.source_id = source_id
        self.quantity = quantity
        self.vlt = vlt
        self.reward_discount = reward_discount


class ManufactureAction(SupplyChainAction):
    def __init__(self, id: int, production_rate: float) -> None:
        super(ManufactureAction, self).__init__(id=id)
        self.production_rate = production_rate
