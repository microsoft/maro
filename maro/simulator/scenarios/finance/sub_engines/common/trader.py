import math
from enum import Enum
from collections import OrderedDict
from maro.simulator.scenarios.finance.common import (
    Action, OrderMode
)


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

    def trade(self, order_action: Action, cur_data: dict, remaining_money: float):
        # return stock, success, stock_price, number, tax
        asset = order_action.item_index
        is_success = False
        is_trigger = True

        deal_volume = abs(order_action.number)
        is_buy = True if order_action.number >= 0 else False

        deal_price = cur_data[order_action.item_index].opening_price
        if 'deal_price' in self._trade_constrain:
            if self._trade_constrain['deal_price'] == 'closing':
                deal_price = cur_data[order_action.item_index].closing_price
        commission_charge = 0

        if order_action.order_mode not in self._order_handlers:
            pass
        else:
            order_handler = self._order_handlers[order_action.order_mode]

            if not order_handler.is_trigger(order_action, cur_data):
                # order not triggered
                is_trigger = False
            else:
                is_success = True
                if self._slippage_handler is not None:
                    deal_price = round(self._slippage_handler.execute(order_action, cur_data, deal_price), 2)

                if is_buy:
                    # if remaining_money is not enough
                    deal_money = deal_price * deal_volume
                    if deal_money > remaining_money:
                        deal_volume -= math.ceil((deal_money - remaining_money) / deal_price)

                    # if min buy unit is configed
                    if TradeConstrain.min_buy_unit.value in self._trade_constrain.keys() \
                            and self._trade_constrain[TradeConstrain.min_buy_unit.value] != 0:
                        odd_volume = deal_volume % self._trade_constrain[TradeConstrain.min_buy_unit.value]
                        if odd_volume != 0:
                            deal_volume -= odd_volume

                else:  # sell
                    avalib_volume = cur_data[order_action.item_index].account_hold_num
                    if avalib_volume < deal_volume:
                        deal_volume = avalib_volume

                    # if min sell unit is configed
                    if TradeConstrain.min_sell_unit.value in self._trade_constrain.keys() \
                            and self._trade_constrain[TradeConstrain.min_sell_unit.value] != 0:
                        odd_volume = abs(deal_volume) % self._trade_constrain[TradeConstrain.min_sell_unit.value]
                        if odd_volume != 0:
                            if odd_volume != avalib_volume % self._trade_constrain[TradeConstrain.min_sell_unit.value]:
                                deal_volume -= odd_volume

                for commission_handler in self._commission_handlers:
                    if is_buy:
                        commission_charge += commission_handler.execute(deal_price, deal_volume)
                    else:
                        commission_charge += commission_handler.execute(deal_price, -deal_volume)

                commission_charge = int(commission_charge * 100) / 100

                # if remaining_money is not enough for commission
                # TODO: sell out money not cover commission not implimented
                if is_buy:
                    expected_remaining = remaining_money - deal_volume * deal_price - commission_charge
                    if expected_remaining < 0:
                        deal_volume -= math.ceil((-expected_remaining) / deal_price)

                        if TradeConstrain.min_buy_unit.value in self._trade_constrain.keys() \
                                and self._trade_constrain[TradeConstrain.min_buy_unit.value] != 0:
                            odd_volume = deal_volume % self._trade_constrain[TradeConstrain.min_buy_unit.value]
                            if odd_volume != 0:
                                deal_volume -= odd_volume

                        if deal_volume < 0:
                            deal_volume = 0
                            is_success = False

                        # recalculate commission charge
                        commission_charge = 0
                        for commission_handler in self._commission_handlers:
                            commission_charge += commission_handler.execute(deal_price, deal_volume)

                self._trade_number += 1

        return asset, is_success, deal_price, deal_volume if is_buy else -deal_volume, commission_charge, is_trigger

    def split_trade(self, order_action: Action, cur_data: dict, remaining_money: float, split_volume: int):
        # return stock, success, stock_price, number, tax
        print("warning: split trading!", "total_volume: ", order_action.number)

        asset = order_action.item_index
        is_success = True
        is_trigger = False
        remaining_volume = abs(order_action.number)
        is_buy = True if order_action.number >= 0 else False
        split_trades = []
        deal_price = cur_data[order_action.item_index].opening_price
        if 'deal_price' in self._trade_constrain:
            if self._trade_constrain['deal_price'] == 'closing':
                deal_price = cur_data[order_action.item_index].closing_price
        continue_splippage_price = deal_price

        print(deal_price)

        while remaining_volume > 0 and is_success:
            deal_volume = remaining_volume if remaining_volume < split_volume else split_volume
            commission_charge = 0

            if order_action.order_mode not in self._order_handlers:
                is_success = False
            else:
                order_handler = self._order_handlers[order_action.order_mode]

                if not order_handler.is_trigger(order_action, cur_data):
                    is_trigger = False
                else:
                    if self._slippage_handler is not None:
                        continue_splippage_price = self._slippage_handler.execute(
                            order_action, cur_data, continue_splippage_price
                        )
                        deal_price = round(continue_splippage_price, 2)

                    if is_buy:
                        # if remaining_money is not enough
                        deal_money = deal_price * deal_volume
                        if deal_money > remaining_money:
                            is_success = False
                            deal_volume -= math.ceil((deal_money - remaining_money) / deal_price)

                        # if min buy unit is configed
                        if TradeConstrain.min_buy_unit.value in self._trade_constrain.keys() \
                                and self._trade_constrain[TradeConstrain.min_buy_unit.value] != 0:
                            odd_volume = deal_volume % self._trade_constrain[TradeConstrain.min_buy_unit.value]
                            if odd_volume != 0:
                                deal_volume -= odd_volume

                    else:
                        avalib_volume = cur_data[order_action.item_index].account_hold_num
                        if avalib_volume < deal_volume:
                            is_success = False
                            deal_volume = avalib_volume

                        # if min sell unit is configed
                        if TradeConstrain.min_sell_unit.value in self._trade_constrain.keys() \
                                and self._trade_constrain[TradeConstrain.min_sell_unit.value] != 0:
                            odd_volume = deal_volume % self._trade_constrain[TradeConstrain.min_sell_unit.value]
                            if odd_volume != 0:
                                if odd_volume != avalib_volume % \
                                        self._trade_constrain[TradeConstrain.min_sell_unit.value]:
                                    deal_volume -= odd_volume

                    if deal_volume <= 0:
                        deal_volume = 0
                        is_success = False

                    for commission_handler in self._commission_handlers:
                        if is_buy:
                            commission_charge += commission_handler.execute(deal_price, deal_volume)
                        else:
                            commission_charge += commission_handler.execute(deal_price, -deal_volume)

                    commission_charge = int(commission_charge * 100) / 100

                    # if remaining_money is not enough for commission
                    # TODO: sell out money not enough for commission not covered
                    if is_buy:
                        expected_remaining = remaining_money - deal_volume * deal_price - commission_charge
                        if expected_remaining < 0:
                            is_success = False
                            deal_volume -= math.ceil((-expected_remaining) / deal_price)

                            if TradeConstrain.min_buy_unit.value in self._trade_constrain.keys() \
                                    and self._trade_constrain[TradeConstrain.min_buy_unit.value] != 0:
                                odd_volume = deal_volume % self._trade_constrain[TradeConstrain.min_buy_unit.value]
                                if odd_volume != 0:
                                    deal_volume -= odd_volume

                            if deal_volume < 0:
                                deal_volume = 0
                                is_success = False

                            # recalculate commission charge
                            commission_charge = 0
                            for commission_handler in self._commission_handlers:
                                commission_charge += commission_handler.execute(deal_price, deal_volume)

                    self._trade_number += 1
                    if is_success:
                        remaining_volume -= deal_volume
                        if is_buy:
                            split_trades.append((deal_price, deal_volume, commission_charge))
                        else:
                            split_trades.append((deal_price, -deal_volume, commission_charge))
                            print(
                                "split trade:", "deal_price:", deal_price, "deal_volume:",
                                deal_volume, "commission_charge:", commission_charge
                            )
                        print(
                            "no.:", self._trade_number, "deal_price: ", deal_price, "deal_volume: ", deal_volume,
                            "commission_charge: ", commission_charge, "remaining_money: ", remaining_money
                        )
                        # print( commission_charge)
        total_amount = 0
        total_volume = 0
        total_commission = 0
        for trade in split_trades:
            total_amount += trade[0] * trade[1]
            total_volume += trade[1]
            total_commission += trade[2]

        return asset, True if total_volume != 0 else False, \
            round(total_amount / total_volume, 2) if total_volume != 0 else 0, \
            total_volume, commission_charge, is_trigger
