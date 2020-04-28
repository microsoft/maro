import math
from enum import Enum
from collections import OrderedDict
from maro.simulator.scenarios.finance.common import (Action, DecisionEvent,
                                                     FinanceType, TradeResult, OrderMode)


class TradeConstrain(Enum):
    min_buy_unit = "min_buy_unit"
    min_sell_unit = "min_sell_unit"


class Trader():
    def __init__(self, trade_constrain: dict):
        self._order_handlers = OrderedDict()
        self._slippage_handler = None
        self._commission_handlers = []
        self._trade_constrain = OrderedDict()
        for constrain in trade_constrain:
            self._trade_constrain[constrain] = trade_constrain[constrain]

        self._trade_number = 0

    @property
    def supported_orders(self) -> list:
        return self._order_handlers.keys()

    def order_handler_register(self, order_type: OrderMode, order_handler: callable):
        self._order_handlers[order_type] = order_handler

    def slippage_handler_register(self, slippage_handler: callable):
        self._slippage_handler = slippage_handler

    def commission_handler_register(self, commission_handler: callable):
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
                    # if remaining_money is not enough
                    actual_money = actual_price * actual_volume
                    if actual_money > remaining_money:
                        actual_volume -= math.ceil((actual_money - remaining_money)/actual_price)

                    # if min buy unit is configed
                    if TradeConstrain.min_buy_unit.value in self._trade_constrain.keys() and self._trade_constrain[TradeConstrain.min_buy_unit.value] != 0:
                        odd_volume = actual_volume % self._trade_constrain[TradeConstrain.min_buy_unit.value]
                        if odd_volume != 0:
                            actual_volume -= odd_volume

                else:
                    remaining_volume = cur_data[order_action.item_index].account_hold_num
                    if remaining_volume < actual_volume:
                        actual_volume = remaining_volume

                    # if min sell unit is configed
                    if TradeConstrain.min_sell_unit.value in self._trade_constrain.keys() and self._trade_constrain[TradeConstrain.min_sell_unit.value] != 0:
                        odd_volume = actual_volume % self._trade_constrain[TradeConstrain.min_sell_unit.value]
                        if odd_volume != 0:
                            if odd_volume != remaining_volume % self._trade_constrain[TradeConstrain.min_sell_unit.value]:
                                actual_volume -= odd_volume

                for commission_handler in self._commission_handlers:
                    commission_charge += commission_handler.execute(actual_price, actual_volume)

                commission_charge = int(commission_charge*100) / 100

                # if remaining_money is not enough for commission
                if actual_volume > 0:
                    expected_remaining = remaining_money - actual_volume * actual_price - commission_charge
                    if expected_remaining < 0:
                        actual_volume -= math.ceil((-expected_remaining)/actual_price)

                        if TradeConstrain.min_buy_unit.value in self._trade_constrain.keys() and self._trade_constrain[TradeConstrain.min_buy_unit.value] != 0:
                            odd_volume = actual_volume % self._trade_constrain[TradeConstrain.min_buy_unit.value]
                            if odd_volume != 0:
                                actual_volume -= odd_volume
                        
                        if actual_volume < 0:
                            actual_volume = 0
                            is_success = False

                        # recalculate commission charge
                        commission_charge = 0
                        for commission_handler in self._commission_handlers:
                            commission_charge += commission_handler.execute(actual_price, actual_volume)

                self._trade_number += 1
                print("no.:", self._trade_number, "actual_price: ", actual_price, "actual_volume: ", actual_volume, "commission_charge: ", commission_charge, "remaining_money: ", remaining_money)
                # print( commission_charge)

        return asset, is_success, actual_price, actual_volume, commission_charge
