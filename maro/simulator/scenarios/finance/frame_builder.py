# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import FrameBase, FrameNode
from maro.simulator.scenarios.finance.account import Account
from maro.simulator.scenarios.finance.stock import Stock


def build_frame(stock_num: int, snapshots_num: int):
    """Function to build finance Frame.

    Args:
        stock_num (int): Number of stocks.
        snapshot_num (int): Number of in-memory snapshots.

    Returns:
        StockFrame: Frame instance for finance scenario.
    """

    class StockFrame(FrameBase):
        stocks = FrameNode(Stock, stock_num)
        account = FrameNode(Account, 1)

        def __init__(self):
            super().__init__(enable_snapshot=True, total_snapshot=snapshots_num)

    return StockFrame()
