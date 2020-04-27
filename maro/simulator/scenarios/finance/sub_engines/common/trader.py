import math
from enum import Enum
from collections import OrderedDict
from maro.simulator.scenarios.finance.common import (Action, DecisionEvent,
                                                     FinanceType, TradeResult)

class TradeConstrain(Enum):
    min_buy_unit = "min_buy_unit"
    min_sell_unit = "min_sell_unit"


class Trader():
    def __init__(self, trade_constrain):
        self._order_handlers = OrderedDict()
        self._slippage_handler = None
        self._commission_handlers = []
        self._trade_constrain = OrderedDict()
        for constrain in trade_constrain:
            self._trade_constrain[constrain] = trade_constrain[constrain]

    @property
    def supported_orders(self):
        return self._order_handlers.keys()

    def order_handler_register(self, order_type, order_handler):
        self._order_handlers[order_type] = order_handler

    def slippage_handler_register(self, slippage_handler):
        self._slippage_handler = slippage_handler

    def commission_handler_register(self, commission_handler):
        self._commission_handlers.append(commission_handler)

    def trade(self, order_action: Action, cur_data: dict, remaining_money: float) -> TradeResult:
        # return stock, success, stock_price, number, tax
        asset = order_action.item_index
        is_success = False
        actual_volume = order_action.number
        actual_price = cur_data[order_action.item_index].closing_price
        commission_charge = 0

        if order_action.order_mode not in self._order_handlers:
            pass
        else:
            order_handler = self._order_handlers[order_action.order_mode]

            if not order_handler.is_trigger(order_action, cur_data):
                pass
            else:
                is_success = True
                if self._slippage_handler is not None:
                    actual_price = self._slippage_handler.execute(order_action, cur_data)

                actual_volume = order_action.number

                if actual_volume >= 0:
                    actual_money = actual_price * actual_volume
                    if actual_money > remaining_money:
                        actual_volume -= math.ceil((actual_money - remaining_money)/actual_price)

                    if TradeConstrain.min_buy_unit.value in self._trade_constrain.keys() and self._trade_constrain[TradeConstrain.min_buy_unit.value] != 0:
                        odd_volume = actual_volume % self._trade_constrain[TradeConstrain.min_buy_unit.value]
                        if odd_volume !=0:
                            actual_volume -= odd_volume

                else:
                    remaining_volume = cur_data[order_action.item_index].account_hold_num
                    if remaining_volume < actual_volume:
                        actual_volume = remaining_volume
                    
                    if TradeConstrain.min_sell_unit.value in self._trade_constrain.keys() and self._trade_constrain[TradeConstrain.min_sell_unit.value] != 0:
                        odd_volume = actual_volume % self._trade_constrain[TradeConstrain.min_sell_unit.value]
                        if odd_volume != 0: 
                            if odd_volume != remaining_volume % self._trade_constrain[TradeConstrain.min_sell_unit.value]:
                                actual_volume -= odd_volume
                for commission_handler in self._commission_handlers:
                    commission_charge += commission_handler.execute(actual_price, actual_volume)

                commission_charge = math.floor(commission_charge*100) /100

                # print("actual_price: ", actual_price, "actual_volume: ", actual_volume, "commission_charge: ", commission_charge)
                print( commission_charge)
                

        return asset, is_success, actual_price, actual_volume, commission_charge
