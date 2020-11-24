# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
from collections import OrderedDict
from typing import List

from yaml import safe_load

from maro.backends.frame import FrameBase, SnapshotList
from maro.data_lib import BinaryReader
from maro.event_buffer import DECISION_EVENT, Event, EventBuffer
from maro.simulator.scenarios import AbsBusinessEngine
from maro.simulator.scenarios.finance.account import Account
from maro.simulator.scenarios.finance.common import (
    Action, ActionState, Cancel, CancelActionScope, CancelDecisionEvent, FinanceEventType, Order, OrderActionScope,
    OrderDecisionEvent, OrderDirection, OrderMode, Trade, two_decimal_price
)
from maro.simulator.scenarios.finance.frame_builder import build_frame
from maro.simulator.scenarios.finance.slippage import FixedSlippage
from maro.simulator.scenarios.finance.stock import Stock
from maro.simulator.scenarios.finance.trade_cost import StockTradeCost
from maro.simulator.scenarios.helpers import DocableDict
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)

METRICS_DESC = """
Finance metrics used to provide statistics information at current point (may be in the middle of a tick).
It contains following keys:

finished_action_number (int): Accumulative number of finished actions until now.
protfile (list): Protfile of each stock.
remaining_cash (float): Remaining cash in the account.
assets_value (float):Current value of the holding assets.
"""
SUPPORTED_ORDER_MODES = [
    OrderMode.MARKET_ORDER,
    OrderMode.STOP_ORDER,
    OrderMode.LIMIT_ORDER,
    OrderMode.STOP_LIMIT_ORDER
]


class FinanceBusinessEngine(AbsBusinessEngine):
    def __init__(
        self, event_buffer: EventBuffer, topology: str, start_tick: int,
        max_tick: int, snapshot_resolution: int, max_snapshot_num: int, additional_options: dict = {}
    ):
        super().__init__(
            scenario_name="finance", event_buffer=event_buffer, topology=topology,
            start_tick=start_tick, max_tick=max_tick, snapshot_resolution=snapshot_resolution,
            max_snapshot_num=max_snapshot_num, additional_options=additional_options
        )

        self._supported_order_mode = SUPPORTED_ORDER_MODES
        # The orders that can be canceled.
        self._pending_orders = OrderedDict()
        self._day_trade_orders = []
        # All the actions finished, including orders, cancel orders, etc.
        self._finished_action = OrderedDict()

        self._load_configs()

        self._load_data()

        self._register_events()

        self._init_frame()

    # Implement common functions.
    @property
    def frame(self) -> FrameBase:
        """FrameBase: Current frame."""
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        """SnapshotList: Current snapshot list."""
        return self._snapshots

    @property
    def configs(self) -> dict:
        """dict: Current configuration."""
        return self._config

    def step(self, tick: int):
        """Push business engine to next step.

        Args:
            tick (int): Current tick to process.
        """

        # Append cancel event.
        decision_event = CancelDecisionEvent(
            tick=tick, action_scope=self._cancel_action_scope(tick)
        )
        event = self._event_buffer.gen_cascade_event(tick, DECISION_EVENT, decision_event)
        self._event_buffer.insert_event(event)

        available_stock_indexes = []

        for stock_index, picker in enumerate(self._stock_pickers):
            for raw_stock in picker.items(tick):
                if raw_stock is not None:
                    # Update stock by code.
                    self._stocks[stock_index].fill(raw_stock)
                    available_stock_indexes.append(stock_index)

        # Append order event.
        for available_stock in available_stock_indexes:
            decision_event = OrderDecisionEvent(
                tick=tick, stock_index=available_stock,
                action_scope=self._order_action_scope(available_stock))
            event = self._event_buffer.gen_cascade_event(tick, DECISION_EVENT, decision_event)

            self._event_buffer.insert_event(event)

    # Functions for generating decision events.
    def _order_action_scope(self, stock_index: int) -> OrderActionScope:
        """Generate the action scope of the order."""
        stock: Stock = self._stocks[stock_index]
        max_sell_volume = 0
        min_sell_volume = 0
        max_buy_volume = 0
        min_buy_volume = 0
        if stock.account_hold_num > 0:
            max_sell_volume = stock.account_hold_num
            odd_shares = stock.account_hold_num % self._trade_unit
            if odd_shares > 0:
                min_sell_volume = odd_shares
            else:
                min_sell_volume = self._trade_unit

        if self._account.remaining_cash > stock.opening_price * self._trade_unit:
            min_buy_volume = self._trade_unit
            max_buy_volume = int(self._account.remaining_cash / stock.opening_price)
            max_buy_volume = max_buy_volume - max_buy_volume % self._trade_unit

        if not self._allow_split:
            max_sell_volume = min(
                max_sell_volume, int(self._max_trade_percent * stock.market_volume)
            )
            min_sell_volume = min(min_sell_volume, max_sell_volume)
            max_buy_volume = min(
                max_buy_volume, int(self._max_trade_percent * stock.market_volume)
            )
            min_buy_volume = min(min_buy_volume, max_buy_volume)

        return OrderActionScope(
            min_buy_volume, max_buy_volume, min_sell_volume, max_sell_volume, self._supported_order_mode
        )

    def _cancel_action_scope(self, tick: int) -> CancelActionScope:
        """Generate the action scope of the cancel action."""
        available_orders = []
        if tick in self._pending_orders:
            available_orders = self._pending_orders[tick]
        return CancelActionScope(available_orders)

    # Implement common functions.
    def post_step(self, tick: int) -> bool:
        """After the events of the tick all finished,
        take the snapshot of the frame and reset the nodes for next tick.

        Args:
            tick (int): Current tick of the step.
        """
        self._handle_day_trading()

        self._update_assets_value()

        # We following the snapshot_resolution settings to take snapshot.
        if (tick + 1) % self._snapshot_resolution == 0:
            # NOTE: We should use frame_index method to get correct index in snapshot list.
            self._frame.take_snapshot(self.frame_index(tick))

        # We reset the stock each tick.
        for stock in self._stocks:
            stock.reset()  # only reset is_valid, sothat the price auto padding

        # Stop current episode if we reach max tick.
        return tick + 1 == self._max_tick

    def get_node_mapping(self) -> dict:
        """Returns: Node mapping of current stations."""
        node_mapping = {}
        for idx, stock in enumerate(self._stocks):
            node_mapping[stock.code] = idx
        return node_mapping

    def reset(self):
        """Reset internal states for episode."""
        self._frame.reset()

        self._snapshots.reset()

        for idx in range(len(self._stock_pickers)):
            self._stock_readers[idx].reset()
            self._stock_pickers[idx] = self._stock_readers[idx].items_tick_picker(
                self._start_tick, self._max_tick, time_unit="d")

        self._account.reset()
        for stock in self._stocks:
            stock.deep_reset()

        self._pending_orders.clear()
        self._day_trade_orders.clear()
        self._finished_action.clear()

    def get_agent_idx_list(self) -> List[int]:
        """Get a list of stock index.

        Returns:
            List[int]: List of stock index.
        """
        return [i for i in range(self._stocks)]

    def get_metrics(self) -> DocableDict:
        """Get current metrics information.

        Note:
            Call this method at different time will get different result.

        Returns:
            DocableDict: Metrics information.
        """
        return DocableDict(
            METRICS_DESC,
            finished_action_number=sum([len(self._finished_action[x]) for x in self._finished_action]),
            protfile=[
                {"code": x.code, "hoding": x.account_hold_num, "average_cost": x.average_cost} for x in self._stocks
            ],
            remaining_cash=self._account.remaining_cash,
            assets_value=self._account.assets_value
        )

    def __del__(self):
        """Collect resource by order."""

        self._stock_pickers.clear()

        for reader in self._stock_readers:
            # Close binary reader first, so that we can clean it correctly.
            reader.close()

    # Customized propertys.
    @property
    def supported_order_modes(self) -> list:
        return self._supported_order_mode

    # Init functions.
    def _load_configs(self):
        """Load configurations."""
        self._update_config_root_path(__file__)
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._config = safe_load(fp)

            # Code of the stocks.
            self._stock_codes = self._config["stocks"]
            # Path of stock data file.
            self._data_path = self._config["data_path"]
            # Initial cash of the account.
            self._init_cash = self._config['account']['init_cash']
            # Amount of the shares pre unit in trading.
            self._trade_unit = self._config["trade_constraint"]["trade_unit"]
            # If env auto split the order into small ones.
            self._allow_split = self._config["trade_constraint"]["allow_split"]
            # The percent of the market volume pre splited order.
            self._split_trade_percent = self._config["trade_constraint"]["split_trade_percent"]
            # The max percent of the market volume the order can be.
            self._max_trade_percent = self._config["trade_constraint"]["max_trade_percent"]
            # The slippage rate of the price when trading.
            self._slippage = self._config["trade_constraint"]["slippage"]
            # The commission rate of the broker.
            self._open_commission = self._config["trade_constraint"]["open_commission"]
            # The commission rate of the broker.
            self._close_commission = self._config["trade_constraint"]["close_commission"]
            # The tax rate of the market.
            self._open_tax = self._config["trade_constraint"]["open_tax"]
            # The tax rate of the market.
            self._close_tax = self._config["trade_constraint"]["close_tax"]
            # The minimum commission of the broker.
            self._min_commission = self._config["trade_constraint"]["min_commission"]
            # If the market allow day trade.
            self._allow_day_trade = self._config["trade_constraint"]["allow_day_trade"]
            # The quote price to chose when maching the order.
            self._deal_price = self._config["trade_constraint"]["deal_price"]

    def _load_data(self):
        self._stock_readers: list = []
        self._stock_pickers: list = []

        # Our stock table used to query stock by tick.
        stock_data_root = self._data_path
        if stock_data_root.startswith("~"):
            stock_data_root = os.path.expanduser(stock_data_root)

        stock_data_paths = []
        for idx in range(len(self._stock_codes)):
            stock_data_paths.append(os.path.join(stock_data_root, f"{self._stock_codes[idx]}.bin"))

        if not os.path.exists(stock_data_root):
            # TODO： implement self._build_temp_data()
            pass

        for idx in range(len(self._stock_codes)):
            self._stock_readers.append(BinaryReader(stock_data_paths[idx]))
            logger.info_green(
                f"Data start date:{self._stock_readers[-1].start_datetime},"
                + f"data end date:{self._stock_readers[-1].end_datetime}"
            )
            self._stock_pickers.append(
                self._stock_readers[idx].items_tick_picker(self._start_tick, self._max_tick, time_unit="d"))

    def _init_frame(self):
        self._frame = build_frame(len(self._stock_codes), self.calc_max_snapshot_num())
        self._snapshots = self._frame.snapshots
        self._account: Account = self._frame.account[0]
        self._account.set_init_state(init_cash=self._init_cash)
        self._stocks = self._frame.stocks

        for idx in range(len(self._stocks)):
            self._stocks[idx].set_init_state(self._stock_codes[idx])

    # Events processing.
    def _register_events(self):
        # Register our own events and their callback handlers.
        self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_recieved)
        self._event_buffer.register_event_handler(FinanceEventType.ORDER, self._take_order)
        self._event_buffer.register_event_handler(FinanceEventType.CANCEL, self._cancel_order)

    def _on_action_recieved(self, event: Event):
        actions = event.payload

        for action in actions:
            if action is None:
                continue
            elif isinstance(action, Cancel):  # Action is Cancel.
                cancel_event = self._event_buffer.gen_atom_event(event.tick, FinanceEventType.CANCEL, action)
                self._event_buffer.insert_event(cancel_event)
            elif isinstance(action, Order):  # Action is Order.
                order_event = self._event_buffer.gen_atom_event(event.tick, FinanceEventType.ORDER, action)
                self._event_buffer.insert_event(order_event)

        # Append panding orders still live but not triggered.
        if event.tick in self._pending_orders:
            for action in self._pending_orders[event.tick]:
                order_event = self._event_buffer.gen_atom_event(event.tick, FinanceEventType.ORDER, action)
                self._event_buffer.insert_event(order_event)
            self._pending_orders[event.tick].clear()

    # Order processing.
    def _take_order(self, event: Event):
        """Match the order in the market and then apply the matching result to account.

        Args:
            order (Order): The order subbmited from agent.
            tick (int): The tick when the order is processed.
        """

        tick = event.tick
        order: Order = event.payload
        if order.state != ActionState.REQUEST:
            # Order is canceled.
            return

        # 1. If the order is triggered.
        # 2. Match trades for the order.
        # 3. Apply the matched trade.
        stock = self._stocks[order.stock_index]
        is_trigger = order.is_trigger(stock.opening_price, stock.market_volume)
        if is_trigger:
            # The order is triggered.
            logger.info_green(f"++++ executing {order}")
            self._execute_order(order=order, tick=tick)
        else:
            # The order is not triggered.
            if order.remaining_life_time > 1:
                self._record_pending_order(order=order, tick=tick)
            else:
                self._finish_action(action=order, tick=tick, state=ActionState.EXPIRED)

    def _execute_order(self, order: Order, tick: int):
        """Match and apply the order.

        Args:
            order (Order): Order to execute.
            tick (int): Current tick of the environment.
        """
        is_matching_success = False
        # Match the order.
        matched_trades = []

        market_price = self._pick_market_price(self._stocks[order.stock_index])
        market_volume = self._stocks[order.stock_index].market_volume

        slippage = FixedSlippage(self._slippage)
        trade_cost = StockTradeCost(
            open_tax=self._open_tax, close_tax=self._close_tax, open_commission=self._open_commission,
            close_commission=self._close_commission, min_commission=self._min_commission
        )

        remaining_volume = order.order_volume
        max_trade_volume = self._max_trade_percent * market_volume
        if remaining_volume > max_trade_volume:
            remaining_volume = max_trade_volume

        remaining_cash = self._account.remaining_cash
        split_volume = self._split_trade_percent * market_volume
        odd_shares = self._stocks[order.stock_index].account_hold_num % self._trade_unit
        matching_price = market_price

        if order.order_direction == OrderDirection.SELL:
            # Holding clipping.
            remaining_volume = min(remaining_volume, self._stocks[order.stock_index].account_hold_num)
            matching = True

            while matching:
                if self._allow_split:
                    # Split clipping.
                    matching_volume = min(remaining_volume, split_volume)
                else:
                    matching_volume = remaining_volume

                adjusted = True
                matching_price_base = matching_price
                # Adjust the volume until match a trade.
                while adjusted:
                    adjusted = False
                    matching_price = matching_price_base

                    matching_price = slippage.calculate(
                        order.order_direction, matching_volume, matching_price, market_volume
                    )

                    matching_cost = trade_cost.calculate(order.order_direction, matching_price, matching_volume)

                    # Trade unit clipping.
                    adjusted, matching_volume, odd_shares = self._adjust_sell_on_trade_unit(matching_volume, odd_shares)
                    if adjusted:
                        continue

                    # Remaining money clipping.
                    adjusted, matching_volume = self._adjust_sell_on_remaining_cash(
                        matching_price, matching_volume, matching_cost, remaining_cash
                    )
                    if adjusted:
                        # Failed to match a selling trade.
                        break

                if not self._allow_split:
                    # If not allow split, match only 1 trade.
                    matching = False

                if matching_volume > 0:
                    # Matched a trade.
                    ret = Trade(
                        trade_direction=order.order_direction, trade_volume=matching_volume,
                        trade_price=matching_price, trade_cost=matching_cost
                    )
                    is_matching_success = True
                    remaining_volume -= matching_volume

                    remaining_cash += ret.trade_volume * ret.trade_price - ret.trade_cost

                    if remaining_volume < 0 or remaining_cash < 0:
                        matching = False
                    else:
                        matched_trades.append(ret)
                else:
                    # Failed to match a trade.
                    matching = False

        elif order.order_direction == OrderDirection.BUY:
            matching = True
            while matching:
                if self._allow_split:
                    # Split clipping.
                    matching_volume = min(remaining_volume, split_volume)
                else:
                    matching_volume = remaining_volume

                adjusted = True
                matching_price_base = matching_price
                while adjusted:
                    adjusted = False
                    matching_price = matching_price_base
                    matching_price = slippage.calculate(
                        order.order_direction, matching_volume, matching_price, market_volume
                    )

                    matching_cost = trade_cost.calculate(order.order_direction, matching_price, matching_volume)

                    # Trade unit clipping.
                    adjusted, matching_volume = self._adjust_buy_on_trade_unit(matching_volume)
                    if adjusted:
                        odd_shares = 0
                        continue

                    # Remaining money clipping.
                    adjusted, matching_volume = self._adjust_buy_on_remaining_cash(
                        matching_price, matching_volume, matching_cost, remaining_cash
                    )
                    if adjusted:
                        continue

                if not self._allow_split:
                    # If not allow split, match only 1 trade.
                    matching = False

                if matching_volume > 0:
                    # Matched a trade.
                    ret = Trade(
                        trade_direction=order.order_direction, trade_volume=matching_volume,
                        trade_price=matching_price, trade_cost=matching_cost
                    )
                    is_matching_success = True
                    remaining_volume -= matching_volume
                    remaining_cash -= ret.trade_volume * ret.trade_price + ret.trade_cost

                    if remaining_volume < 0 or remaining_cash < 0:
                        matching = False
                    else:
                        matched_trades.append(ret)
                else:
                    # Failed to match a trade.
                    matching = False

        # Apply trade.
        if is_matching_success:
            for matched_trade in matched_trades:
                logger.info_green(f"++++ trading: {matched_trade}")
                if order.order_direction == OrderDirection.BUY:
                    self._stocks[order.stock_index].average_cost = (
                        self._stocks[order.stock_index].account_hold_num
                        * self._stocks[order.stock_index].average_cost
                        + matched_trade.trade_price * matched_trade.trade_volume
                    ) / (self._stocks[order.stock_index].account_hold_num + matched_trade.trade_volume)
                    # If day trading is allowed, apply the new buy-in volume delta to the stock immediately.
                    if self._allow_day_trade:
                        self._stocks[order.stock_index].account_hold_num += matched_trade.trade_volume
                    else:
                        if order not in self._day_trade_orders:
                            self._day_trade_orders.append(order)
                else:
                    self._stocks[order.stock_index].account_hold_num -= matched_trade.trade_volume

                # Apply trade cash delta in account.
                self._account.remaining_cash += two_decimal_price(matched_trade.cash_delta)
            order.state = ActionState.FINISH
            self._finish_action(action=order, tick=tick, state=ActionState.FINISH, result=matched_trades)
        else:
            self._finish_action(
                action=order, tick=tick, state=ActionState.FAILED, comment="Can not match any trade for the order."
            )

    def _adjust_buy_on_trade_unit(self, matching_volume: int):
        adjusted = False
        adjusted_volume = matching_volume
        if matching_volume % self._trade_unit != 0:
            adjusted = True
            odd_volume = adjusted_volume % self._trade_unit
            adjusted_volume -= odd_volume
        return adjusted, adjusted_volume

    def _adjust_sell_on_trade_unit(self, matching_volume: int, odd_shares: int):
        adjusted = False
        adjusted_volume = matching_volume
        odd_volume = matching_volume % self._trade_unit
        if odd_volume != 0:
            if odd_shares != odd_volume:
                adjusted = True
                adjusted_volume -= odd_volume
                adjusted_volume += odd_shares
                odd_shares = 0
        return adjusted, adjusted_volume, odd_shares

    def _adjust_buy_on_remaining_cash(
        self, matching_price: float,
        matching_volume: int, matching_cost: float, remaining_cash: float
    ):
        adjusted = False
        adjusted_volume = matching_volume
        money_needed = matching_price * matching_volume + matching_cost
        if money_needed > remaining_cash:
            adjusted = True
            adjusted_volume = adjusted_volume - math.ceil(
                (money_needed - remaining_cash) / (matching_price * self._trade_unit)
            ) * self._trade_unit

        return adjusted, int(adjusted_volume)

    def _adjust_sell_on_remaining_cash(
        self, matching_price: float,
        matching_volume: int, matching_cost: float, remaining_cash: float
    ):
        adjusted = False
        adjusted_volume = matching_volume
        money_needed = matching_cost
        if money_needed > remaining_cash + matching_price * matching_volume:
            # Shall failed.
            adjusted = True
            adjusted_volume = 0

        return adjusted, int(adjusted_volume)

    # Cancel processing.
    def _cancel_order(self, event: Event):
        """Cancel the specified order."""
        tick = event.tick
        action: Cancel = event.payload
        action.order.state = ActionState.CANCEL
        self._finish_action(action=action.order, tick=tick)

        action.state = ActionState.FINISH
        self._finish_action(action=action, tick=tick)

    # Extra bussiness logic.
    def _pick_market_price(self, stock: Stock) -> float:
        """Pick market price from quote according to config."""
        return getattr(stock, self._deal_price, stock.opening_price)

    def _is_market_closed(self) -> bool:
        # TODO: Implement
        return True

    def _finish_action(
        self, action: Action, tick: int, state: ActionState = ActionState.FINISH,
        result: any = None, comment: str = None
    ):
        action.finish(tick=tick, state=state, result=result, comment=comment)
        if tick not in self._finished_action:
            self._finished_action[tick] = []
        self._finished_action[tick].append(action)

    def _record_pending_order(self, order: Order, tick: int):
        order.remaining_life_time -= 1
        if tick + 1 not in self._pending_orders:
            self._pending_orders[tick + 1] = []
        self._pending_orders[tick + 1].append(order)

    def _update_assets_value(self):
        assets_value = 0
        for stock in self._stocks:
            assets_value += stock.closing_price * stock.account_hold_num
        self._account.assets_value = two_decimal_price(assets_value)

    def _handle_day_trading(self):
        # If day trading is not allowed, apply the new buy-in volume delta to the stock at the end of the day.
        if (not self._allow_day_trade) and self._is_market_closed():
            for order in self._day_trade_orders:
                for result in order.action_result:
                    self._stocks[order.stock_index].account_hold_num += result.trade_volume
            self._day_trade_orders.clear()
