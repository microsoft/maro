from maro.simulator.scenarios.finance.common.common import OrderMode


class Order():
    __order_type = None

    def __init__(self):
        pass

    def is_trigger(self, order_action: OrderMode, cur_data: dict):
        pass

    @property
    def order_type(self):
        return self.__order_type

    @order_type.setter
    def order_type(self, value):
        self.__order_type = value


class MarketOrder(Order):
    def __init__(self):
        Order.__init__(self)
        self.order_type = OrderMode.market_order

    def is_trigger(self, order_action: OrderMode, cur_data: dict) -> bool:
        triggered = False
        if cur_data[order_action.item_index].trade_volume > 0:
            triggered = True
        print(f'Market Order triggered: {triggered}')
        return triggered


class LimitOrder(Order):
    def __init__(self):
        Order.__init__(self)
        self.order_type = OrderMode.limit_order

    def is_trigger(self, order_action: OrderMode, cur_data: dict) -> bool:
        triggered = False
        if cur_data[order_action.item_index].trade_volume > 0:
            if order_action.number >= 0:  # buy
                if order_action.limit > cur_data[order_action.item_index].opening_price:
                    triggered = True
            else:  # sell
                if order_action.limit < cur_data[order_action.item_index].opening_price:
                    triggered = True
        print(f'Limit Order triggered: {triggered}')
        return triggered


class StopOrder(Order):
    def __init__(self):
        Order.__init__(self)
        self.order_type = OrderMode.stop_order

    def is_trigger(self, order_action: OrderMode, cur_data: dict) -> bool:
        triggered = False
        if cur_data[order_action.item_index].trade_volume > 0:
            if order_action.number >= 0:  # buy
                if order_action.stop < cur_data[order_action.item_index].opening_price:
                    triggered = True
            else:  # sell
                if order_action.stop > cur_data[order_action.item_index].opening_price:
                    triggered = True
        print(f'Stop Order triggered: {triggered}')
        return triggered


class StopLimitOrder(Order):
    def __init__(self):
        Order.__init__(self)
        self.order_type = OrderMode.stop_limit_order

    def is_trigger(self, order_action: OrderMode, cur_data: dict) -> bool:
        triggered = False
        if cur_data[order_action.item_index].trade_volume > 0:
            if order_action.number >= 0:  # buy
                if order_action.stop < cur_data[order_action.item_index].opening_price:
                    if order_action.limit > cur_data[order_action.item_index].opening_price:
                        triggered = True
            else:  # sell
                if order_action.stop > cur_data[order_action.item_index].opening_price:
                    if order_action.limit < cur_data[order_action.item_index].opening_price:
                        triggered = True

        print(f'Stop Limit Order triggered: {triggered}')
        return triggered
