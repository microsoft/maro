# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator.scenarios.supply_chain import SellerUnit


class SimpleSellerUnit(SellerUnit):
    def market_demand(self, tick: int) -> int:
        return tick
