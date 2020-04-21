from collections import OrderedDict
from maro.simulator.scenarios.finance.common import (Action, DecisionEvent,
                                                     FinanceType, TradeResult)

class Trader():    
    def __init__(self):
        self._order_handlers = OrderedDict()
        self._slippage_handler = None
        self._commission_handlers = []

    @property
    def supported_orders(self):
        return self._order_handlers.keys()

    def order_handler_register(self, order_type, order_handler):
        self._order_handlers[order_type] = order_handler

    def slippage_handler_register(self, slippage_handler):
        self._slippage_handler=slippage_handler

    def commission_handler_register(self, commission_handler):
        self._commission_handlers.append(commission_handler)

    def trade(self, order_action: Action, cur_data: dict) -> TradeResult:
        # return stock, success, stock_price, number, tax
        asset = order_action.item_index
        is_success = False
        actual_volume = order_action.number
        actual_price = cur_data[order_action.item_index].closing_price
        commission_charge = 0

        if order_action.order_type not in self._order_handlers:
            pass
        else:
            order_handler = self._order_handlers[order_action.order_type]
        
            if not order_handler.is_trigger(order_action, cur_data):
                pass
            else:
                is_success = True
                if self._slippage_handler is not None:
                    actual_price, actual_volume = self._slippage_handler.execute(order_action, cur_data)
                
                for commission_handler in self._commission_handlers:
                    commission_charge += commission_handler.execute(order_action, actual_price, actual_volume)

        return asset, is_success, actual_price, actual_volume, commission_charge