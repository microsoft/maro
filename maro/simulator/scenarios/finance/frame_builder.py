# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import FrameNode, FrameBase

from .stock.stock import Stock
from maro.simulator.scenarios.finance.account import Account


def build_frame(stock_num: int, snapshots_num: int):
    """Function to build citi_bike Frame.

    Args:
        station_num (int): Number of stations.
        snapshot_num (int): Number of in-memory snapshots.

    Returns:
        CitibikeFrame: Frame instance for citi-bike scenario.
    """

    class StockFrame(FrameBase):
        stocks = FrameNode(Stock, stock_num)
        account = FrameNode(Account, 1)

        def __init__(self):
            super().__init__(enable_snapshot=True, total_snapshot=snapshots_num)

    return StockFrame()
