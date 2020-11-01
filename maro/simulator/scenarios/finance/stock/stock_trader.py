from collections import OrderedDict
from maro.simulator.scenarios.finance.common.trader import Trader
from maro.simulator.scenarios.finance.common.common import (OrderMode)
from maro.simulator.scenarios.finance.common.order import MarketOrder, LimitOrder, StopOrder, StopLimitOrder
from maro.simulator.scenarios.finance.common.slippage import ByMoneySlippage
from maro.simulator.scenarios.finance.common.commission import ByMoneyCommission, StampTaxCommission

# decision_event = DecisionEvent(tick, FinanceType.stock, valid_stocks, self.name, self._action_scope)


class StockTrader(Trader):

    def __init__(self, trade_constraint: OrderedDict):
        Trader.__init__(self, trade_constraint)

        self.order_handler_register(OrderMode.market_order, MarketOrder())
        self.order_handler_register(OrderMode.limit_order, LimitOrder())
        self.order_handler_register(OrderMode.stop_order, StopOrder())
        self.order_handler_register(OrderMode.stop_limit_order, StopLimitOrder())

        self.slippage_handler_register(ByMoneySlippage(trade_constraint['slippage']))
        self.commission_handler_register(ByMoneyCommission(trade_constraint['commission'], 5))
        self.commission_handler_register(StampTaxCommission(trade_constraint['close_tax']))
