import os
from typing import Dict, List
from maro.simulator.scenarios.modelbase import build_frame
from maro.simulator.event_buffer import Event, EventBuffer
from maro.simulator.frame import Frame, SnapshotList
from maro.simulator.scenarios.finance.abs_sub_business_engine import \
    AbsSubBusinessEngine
from maro.simulator.scenarios.finance.common import FinanceType
from maro.simulator.scenarios.finance.reader import (FinanceDataType,
                                                     FinanceReader)
from maro.simulator.scenarios.finance.reader import Stock as RawStock
from maro.simulator.utils.common import tick_to_frame_index
from maro.simulator.scenarios.finance.common import Action
from .stock import Stock


class StockBusinessEngine(AbsSubBusinessEngine):
    def __init__(self, start_tick: int, max_tick: int, frame_resolution: int, config: dict, event_buffer: EventBuffer):
        super().__init__(start_tick, max_tick, frame_resolution, config, event_buffer)

        self._stock_codes: list = None
        self._stocks: Dict[Stock] = None
        self._readers: dict = None

        self._init_reader()

    @property
    def finance_type(self):
        return FinanceType.stock

    @property
    def frame(self):
        return self._frame

    @property
    def snapshot_list(self): 
        return self._snapshots
    
    @property
    def name_mapping(self):
        return {stock.index: code for code, stock in self._stocks.items()}

    def step(self, tick: int):
        for code, reader in self._readers.items():
            raw_stock: RawStock = reader.next_item()

            if raw_stock is not None:
                # update frame by code
                stock: Stock = self._stocks[code]

                stock.opening_price = raw_stock.opening_price
                stock.closing_price = raw_stock.closing_price
                stock.daily_return = raw_stock.daily_return
                stock.highest_price = raw_stock.daily_return
                stock.lowest_price = raw_stock.lowest_price
                stock.trade_amount = raw_stock.trade_amount
                stock.trade_num = raw_stock.trade_num
                stock.trade_volume = raw_stock.trade_volume

    def post_step(self, tick: int):
        self.snapshot_list.insert_snapshot(self._frame, tick)

    def post_init(self, max_tick: int):
        self._init_frame()
        self._build_stocks()

    def take_action(self, action: Action):
        pass

    def reset(self):
        pass

    def _init_frame(self):
        self._frame = build_frame(Stock, len(self._stock_codes))
        self._snapshots = SnapshotList(self._frame, self._max_tick)

    def _build_stocks(self):
        self._stocks = {}

        for index, code in enumerate(self._stock_codes):
            self._stocks[code] = Stock(self._frame, index, code)

    def _init_reader(self):
        data_folder = self._config["data_path"]

        self._stock_codes = self._config["stocks"]

        # TODO: is it a good idea to open lot of file at same time?
        self._readers = {}

        for code in self._stock_codes:
            data_path = os.path.join(data_folder, f"{code}.bin").encode()

            self._readers[code] = FinanceReader(FinanceDataType.STOCK, data_path, self._start_tick, self._max_tick)

            # in case the data file contains different ticks
            new_max_tick = self._readers[code].max_tick
            self._max_tick = new_max_tick if self._max_tick <=0 else min(new_max_tick, self._max_tick)
