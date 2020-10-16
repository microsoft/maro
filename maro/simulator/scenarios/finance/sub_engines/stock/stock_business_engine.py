import os
from typing import Dict, List
from collections import OrderedDict

from maro.event_buffer import Event, EventBuffer
from maro.simulator.scenarios.finance.abs_sub_business_engine import \
    AbsSubBusinessEngine
from maro.simulator.scenarios.finance.common import (Action, DecisionEvent,
                                                     FinanceType, TradeResult, OrderMode, ActionType, ActionState)

from maro.data_lib import BinaryReader
from maro.simulator.utils.common import tick_to_frame_index
from .frame_builder import build_frame

from maro.simulator.scenarios.finance.account import SubAccount

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
        self._item_picker = None
        self._account: SubAccount = None

        self._action_scope_max = self._config["action_scope"]["max"]

        self._init_reader()
        self._init_trader(self._config)

        self._pending_orders = []
        self._canceled_orders = []

        self._init_frame()
        self._build_stocks()

    @property
    def finance_type(self):
        return FinanceType.stock

    @property
    def account(self):
        return self._account

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
            #raw_stock = reader.next_item()
            for raw_stock in self._item_picker[code].items(tick):
                if raw_stock is not None:
                    #print(raw_stock)
                    # update frame by code
                    stock: Stock = self._stocks_dict[code]
                    stock.fill(raw_stock)
                    #print("tick:", tick,"code:", code, "raw_stock:", raw_stock)
                    if stock.is_valid:
                        valid_stocks.append(code)

        # append cancel_order event
        decision_event = DecisionEvent(tick, FinanceType.stock,-2, self.name, self._action_scope, action_type = ActionType.cancel_order)
        evt = self._event_buffer.gen_cascade_event(tick, DecisionEvent, decision_event)
        self._event_buffer.insert_event(evt)
        # append account event
        decision_event = DecisionEvent(tick, FinanceType.stock,-1, self.name, self._action_scope, action_type = ActionType.transfer)
        evt = self._event_buffer.gen_cascade_event(tick, DecisionEvent, decision_event)
        self._event_buffer.insert_event(evt)
        # append order event
        for valid_stock in valid_stocks:
            decision_event = DecisionEvent(tick, FinanceType.stock, valid_stock, self.name, self._action_scope,action_type = ActionType.order)
            evt = self._event_buffer.gen_cascade_event(tick, DecisionEvent, decision_event)

            self._event_buffer.insert_event(evt)

    def post_step(self, tick: int):
        # after take snapshot, we need to reset the stock
        self.frame.take_snapshot(tick)

        for stock in self._stock_list:
            stock.reset()
        

    def take_action(self, action: Action, remaining_money: float, tick: int) -> TradeResult:
        # 1. can trade -> bool
        # 2. return (stock, sell/busy, stock_price, number, tax)
        # 3. update stock.account_hold_num
        ret = TradeResult(self.name, action.item_index, 0, tick, 0, 0, False, False)
        if action.id not in self._canceled_orders:
            # not canceled
            out_of_scope, allow_split = self._verify_action(action)
            if not out_of_scope:
                if not allow_split:
                    asset, is_success, actual_price, actual_volume, commission_charge, is_trigger = self._trader.trade(action, self._stock_list, remaining_money)  # list  index is in action # self.snapshot
                
                elif allow_split:
                    asset, is_success, actual_price, actual_volume, commission_charge, is_trigger = self._trader.split_trade(
                        action, self._stock_list, remaining_money, self._stock_list[action.item_index].trade_volume * self._action_scope_max)  # list  index is in action # self.snapshot
                ret = TradeResult(self.name, action.item_index, actual_volume, tick, actual_price, commission_charge, is_success, is_trigger)
                if not is_trigger:
                    if action.life_time != 1:
                        action.life_time -= 1
                        self._event_buffer.gen_atom_event(tick+1, DecisionEvent, action)
                        if action.id not in self._pending_orders:
                            self._pending_orders.append(action.id)
                    else:
                        if action.id in self._pending_orders:
                            self._pending_orders.remove(action.id)
                            action.state = ActionState.expired
                            action.finish_tick = tick
                else:
                    if action.id in self._pending_orders:
                        self._pending_orders.remove(action.id)
                    if is_success:
                        action.state = ActionState.success
                        if actual_volume > 0:
                            self._stock_list[asset].average_cost = ((self._stock_list[asset].account_hold_num*self._stock_list[asset].average_cost) +
                                                                    (actual_price*actual_volume))/(self._stock_list[asset].account_hold_num + actual_volume)
                        self._stock_list[asset].account_hold_num += actual_volume
                    else:
                        action.state = ActionState.failed
                    action.finish_tick = tick
                    action.action_result = ret
            else:
                print("Warning: out of action scope and not allow split!", self.name, self._action_scope(action.action_type, action.item_index), action.number)
                action.state = ActionState.failed
                action.finish_tick = tick

            
            
        else:
            # order canceled
            self._canceled_orders.remove(action.id)
            action.state = ActionState.canceled
            action.finish_tick = tick
        print("result", ret)
        return ret

    def cancel_order(self, action: Action):
        if action.id in self._pending_orders:
            self._pending_orders.remove(action.id)
        if action.id not in self._canceled_orders:
            print(f'Order canceled :{action.id}')
            self._canceled_orders.append(action.id)

    def reset(self):
        for _, reader in self._readers.items():
            reader.reset()

        self._frame.reset()
        self._snapshots.reset()
        self._pending_orders = []
        self._canceled_orders = []

    def _action_scope(self, action_type: ActionType, stock_index: int):
        if action_type == ActionType.order:
            # action_scope of stock
            stock: Stock = self._stock_list[stock_index]
            if self._allow_split:
                result = (-stock.account_hold_num, stock.trade_volume)
            else:
                result = (max([-stock.account_hold_num, -stock.trade_volume * self._action_scope_max]), stock.trade_volume * self._action_scope_max)
            return (result, self._trader.supported_orders, self._order_mode)

        elif action_type == ActionType.transfer:
            # action_scope of account
            result = (-self._account.remaining_money, float("inf"))
            return result

        elif action_type == ActionType.cancel_order:
            # action_scope of pending orders
            result = self._pending_orders

            return  result

    def _init_frame(self):
        self._frame = build_frame(len(self._stock_codes), self._max_tick)
        self._snapshots = self._frame.snapshots
        for index in range(len(self._stock_codes)):
            stock = self._frame.stocks[index]
            stock.set_init_state(self._stock_codes[index])

    def _build_stocks(self):
        self._stocks_dict = {}
        self._stock_list = self._frame.stocks

        for index, stock in enumerate(self._stock_list):
            self._stocks_dict[index] = stock

        self._account = self._frame.sub_account[0]
        self._account.set_init_state(currency=self._config['currency'], leverage=self._config['leverage'], min_leverage_rate=self._config['min_leverage_rate'], init_money=self._config["money"])  # we only one account, so the index is 0

    def _init_reader(self):
        data_folder = self._config["data_path"]

        self._stock_codes = self._config["stocks"]

        # TODO: is it a good idea to open lot of file at same time?
        self._readers = {}
        self._item_picker = {}

        for index, code in enumerate(self._stock_codes):
            data_path = os.path.join(data_folder, f"{code}.bin")

            self._readers[index] = BinaryReader(data_path)
            self._item_picker[index] = self._readers[index].items_tick_picker(self._start_tick, self._max_tick, time_unit="d")

            # in case the data file contains different ticks
            # new_max_tick = self._readers[code].max_tick
            # self._max_tick = new_max_tick if self._max_tick <= 0 else min(new_max_tick, self._max_tick)

    def _init_trader(self, config):
        trade_constrain = config['trade_constrain']

        self._trader = StockTrader(trade_constrain)

    def _verify_action(self, action: Action):
        ret = True
        allow_split = self._allow_split
        if self._action_scope(action.action_type, action.item_index)[0][0] <= action.number and self._action_scope(action.action_type, action.item_index)[0][1] >= action.number:
            ret = False
        return ret, allow_split

    @property
    def _allow_split(self):
        allow_split = False
        if "allow_split" in self._config["trade_constrain"]:
            allow_split = self._config["trade_constrain"]["allow_split"]
        return allow_split
