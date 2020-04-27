import os
from typing import Dict, List
from collections import OrderedDict

from maro.simulator.event_buffer import Event, EventBuffer
from maro.simulator.frame import Frame, SnapshotList
from maro.simulator.scenarios.finance.abs_sub_business_engine import \
    AbsSubBusinessEngine
from maro.simulator.scenarios.finance.common import (Action, DecisionEvent,
                                                     FinanceType, TradeResult, OrderMode)
from maro.simulator.scenarios.finance.reader import (FinanceDataType,
                                                     FinanceReader)
from maro.simulator.scenarios.finance.reader import Stock as RawStock
from maro.simulator.scenarios.entity_base import FrameBuilder
from maro.simulator.utils.common import tick_to_frame_index

from .stock import Stock
from .stock_trader import StockTrader
from ..common.trader import TradeConstrain


class StockBusinessEngine(AbsSubBusinessEngine):
    def __init__(self, beginning_timestamp: int, start_tick: int, max_tick: int, frame_resolution: int, config: dict, event_buffer: EventBuffer):
        super().__init__(beginning_timestamp, start_tick, max_tick, frame_resolution, config, event_buffer)

        self._stock_codes: list = None
        self._stocks_dict: dict = None
        self._stock_list: list = None
        self._readers: dict = None
        self._order_mode = OrderMode.market_order
        self._trader = None

        self._action_scope_min = self._config["action_scope"]["min"]
        self._action_scope_max = self._config["action_scope"]["max"]

        self._init_reader()
        self._init_trader(self._config)

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
        return {stock.index: code for code, stock in self._stocks_dict.items()}

    def step(self, tick: int):
        valid_stocks = []

        for code, reader in self._readers.items():
            raw_stock: RawStock = reader.next_item()

            if raw_stock is not None:
                # update frame by code
                stock: Stock = self._stocks_dict[code]
                stock.fill(raw_stock)

                if raw_stock.is_valid:
                    valid_stocks.append(stock.index)

        decision_event = DecisionEvent(tick, FinanceType.stock, valid_stocks, self.name, self._action_scope)
        evt = self._event_buffer.gen_cascade_event(tick, DecisionEvent, decision_event)

        self._event_buffer.insert_event(evt)

    def post_step(self, tick: int):
        # after take snapshot, we need to reset the stock
        self.snapshot_list.insert_snapshot(tick)

        for stock in self._stock_list:
            stock.reset()

    def post_init(self, max_tick: int):
        self._init_frame()
        self._build_stocks()

    def take_action(self, action: Action, remaining_money: float, tick: int) -> TradeResult:
        # 1. can trade -> bool
        # 2. return (stock, sell/busy, stock_price, number, tax)
        # 3. update stock.account_hold_num
        asset, is_success, actual_price, actual_volume, commission_charge = self._trader.trade(action, self._stock_list, remaining_money)  # list  index is in action # self.snapshot
        ret = TradeResult(self.name, action.item_index, actual_volume, tick, actual_price, commission_charge, is_success)
        if is_success:
            self._stock_list[asset].account_hold_num += actual_volume
        return ret

    def reset(self):
        pass

    def _action_scope(self, stock_index_list: list):
        result = {}

        for stock_index in stock_index_list:
            stock: Stock = self._stock_list[stock_index_list]

            result[stock_index] = (stock.trade_volume * self._action_scope_min, stock.trade_volume * self._action_scope_max)

        return (self._order_mode, result, self._trader.supported_orders)

    def _init_frame(self):
        self._frame = FrameBuilder.new().add_model(Stock, len(self._stock_codes)).build()
        self._snapshots = SnapshotList(self._frame, self._max_tick)

    def _build_stocks(self):
        self._stocks_dict = {}
        self._stock_list = []

        for index, code in enumerate(self._stock_codes):
            stock = Stock(self._frame, index, code)
            self._stocks_dict[code] = stock
            self._stock_list.append(stock)

    def _init_reader(self):
        data_folder = self._config["data_path"]

        self._stock_codes = self._config["stocks"]

        # TODO: is it a good idea to open lot of file at same time?
        self._readers = {}

        for code in self._stock_codes:
            data_path = os.path.join(data_folder, f"{code}.bin").encode()

            self._readers[code] = FinanceReader(FinanceDataType.STOCK, data_path, self._start_tick, self._max_tick, self._beginning_timestamp)

            # in case the data file contains different ticks
            new_max_tick = self._readers[code].max_tick
            self._max_tick = new_max_tick if self._max_tick <= 0 else min(new_max_tick, self._max_tick)

    def _init_trader(self, config):
        trade_constrain = OrderedDict()
        trade_constrain[TradeConstrain.min_buy_unit] = config['trade_constrain']['min_buy_unit']
        trade_constrain[TradeConstrain.min_sell_unit] = config['trade_constrain']['min_sell_unit']
        self._trader = StockTrader(trade_constrain)
