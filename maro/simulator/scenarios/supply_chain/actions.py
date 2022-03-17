# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from dataclasses import dataclass


@dataclass
class SupplyChainAction:
    id: int


@dataclass
class ConsumerAction(SupplyChainAction):
    product_id: int
    source_id: int
    quantity: int
    vlt: int  # TODO: update vlt related code


@dataclass
class ManufactureAction(SupplyChainAction):
    production_rate: float
