# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import datetime
import math
import os
from collections import OrderedDict
from typing import List

from dateutil.relativedelta import relativedelta
from dateutil.tz import gettz
from yaml import safe_load

from maro.backends.frame import FrameBase, SnapshotList
from maro.data_lib import BinaryReader
from maro.event_buffer import DECISION_EVENT, Event, EventBuffer
from maro.simulator.scenarios import AbsBusinessEngine
from maro.simulator.scenarios.finance.account import Account
from maro.simulator.scenarios.finance.common.commission import ByMoneyCommission, StampTaxCommission
from maro.simulator.scenarios.finance.common.common import (
    ActionState, ActionType, CancelOrder, CancelOrderActionScope, DecisionEvent, Order,
    OrderActionScope, OrderDirection, OrderMode, TradeResult
)
from maro.simulator.scenarios.finance.common.slippage import ByMoneySlippage
from maro.simulator.scenarios.finance.frame_builder import build_frame
from maro.simulator.scenarios.finance.stock.stock import Stock
from maro.simulator.scenarios.helpers import DocableDict
from maro.utils.logger import CliLogger

LOGGER = CliLogger(name=__name__)

METRICS_DESC = """
"""


class FinanceBusinessEngine(AbsBusinessEngine):
    def __init__(
        self, event_buffer: EventBuffer, topology: str, start_tick: int,
        max_tick: int, snapshot_resolution: int, max_snapshots: int, additional_options: dict = {}
    ):
        super().__init__(
            scenario_name="finance", event_buffer=event_buffer, topology=topology, start_tick=start_tick, max_tick=max_tick,
            snapshot_resolution=snapshot_resolution, max_snapshots=max_snapshots, additional_options=additional_options
        )
        # 1. load_config
        # Update self._config_path with current file path.
        self.update_config_root_path(__file__)
        self._load_configs()
        self._time_zone = gettz(self._conf["time_zone"])

        # 2. load_data
        # Trip binary reader.
        self._quote_readers: list = []
        self._quote_pickers: list = []

        # Our quote table used to query quote by tick.
        quote_data_path = self._conf["data_path"]
        if quote_data_path.startswith("~"):
            quote_data_path = os.path.expanduser(quote_data_path)

        quote_data_paths = []
        for idx in range(len(self._conf["stocks"])):
            quote_data_paths.append(os.path.join(quote_data_path, f"{self._conf['stocks'][idx]}.bin"))

        all_quote_exists = True
        for data_path in quote_data_paths:
            if not os.path.exists(data_path):
                all_quote_exists = False
                break
        if not all_quote_exists:
            self._build_temp_data()

        for idx in range(len(self._conf["stocks"])):
            self._quote_readers.append(BinaryReader(quote_data_paths[idx]))
            print(type(self._quote_readers[-1].start_datetime), self._quote_readers[-1].end_datetime)
            self._quote_pickers.append(
                self._quote_readers[idx].items_tick_picker(self._start_tick, self._max_tick, time_unit="d"))

        # We keep this used to calculate real datetime to get quote info.
        self._quote_start_date: datetime.datetime = datetime.datetime.fromisoformat(self._conf["beginning_date"])
        self._quote_end_date: datetime.datetime = datetime.datetime.fromisoformat(self._conf["ending_date"])

        # Since binary data hold UTC timestamp, convert it into our target timezone.
        self._quote_start_date = self._quote_start_date.astimezone(self._time_zone)
        self._quote_end_date = self._quote_start_date.astimezone(self._time_zone)

        # 3. init_instance
        # Time zone we used to transfer UTC to target time zone.
        self._register_events()

        self._init_frame()

        self._support_order_mode = [
            OrderMode.market_order, OrderMode.stop_order,
            OrderMode.limit_order, OrderMode.stop_limit_order
        ]

        self._pending_orders = OrderedDict()  # The orders that can be canceled.
        self._day_trade_orders = []  # The order list for dealing with day trading
        self._finished_action = OrderedDict()  # All the actions finished, including orders, cancel orders, etc.

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
        return self._conf

    @property
    def supported_order_modes(self) -> list:
        return self._support_order_mode

    def step(self, tick: int):
        """Push business engine to next step.

        Args:
            tick (int): Current tick to process.
        """
        valid_stocks = []

        for idx, picker in zip(range(len(self._quote_pickers)), self._quote_pickers):
            for raw_stock in picker.items(tick):
                if raw_stock is not None:
                    # update frame by code
                    self._stocks[idx].fill(raw_stock)
                    valid_stocks.append(idx)

        # append cancel_order event
        decision_event = DecisionEvent(
            tick, action_scope_func=self._action_scope, action_type=ActionType.cancel_order
        )
        evt = self._event_buffer.gen_cascade_event(tick, DecisionEvent, decision_event)
        self._event_buffer.insert_event(evt)

        # append order event
        for valid_stock in valid_stocks:
            decision_event = DecisionEvent(
                tick, item=valid_stock,
                action_scope_func=self._action_scope, action_type=ActionType.order)
            evt = self._event_buffer.gen_cascade_event(tick, DecisionEvent, decision_event)

            self._event_buffer.insert_event(evt)

    def post_step(self, tick: int) -> bool:
        """After the events of the tick all finished,
        take the snapshot of the frame and reset the nodes for next tick.

        Args:
            tick (int): Current tick of the step.
        """
        if (not self._conf["trade_constraint"]["allow_day_trade"]) and self.is_market_closed():
            # not allowed day trade, add stock buy amount to hold here
            if len(self._day_trade_orders) > 0:
                for order in self._day_trade_orders:
                    if order.direction == OrderDirection.buy:
                        if order.action_result is not None:
                            for result in order.action_result:
                                self._stocks[order.item].account_hold_num += result.trade_number
                self._day_trade_orders.clear()
        self._account.update_assets_value(self._stocks)
        # We following the snapshot_resolution settings to take snapshot.
        if (tick + 1) % self._snapshot_resolution == 0:
            # NOTE: We should use frame_index method to get correct index in snapshot list.
            self._frame.take_snapshot(self.frame_index(tick))

        # We reset the station station each resolution.
        for stock in self._stocks:
            stock.reset()

        # Stop current episode if we reach max tick.
        return tick + 1 == self._max_tick

    def get_node_mapping(self) -> dict:
        """dict: Node mapping of current stations."""
        node_mapping = {}
        for idx, stock in zip(range(len(self._stocks)), self._stocks):
            node_mapping[stock.code] = idx
        return node_mapping

    def reset(self):
        """Reset internal states for episode."""
        self._frame.reset()

        self._snapshots.reset()

        for reader in self._quote_readers:
            reader.reset()

        for idx in range(len(self._quote_pickers)):
            self._quote_pickers[idx] = self._quote_readers[idx].items_tick_picker(
                self._start_tick, self._max_tick, time_unit="d")
        self._account.reset()
        self._account.set_init_state(init_money=self._conf['account']['money'])
        for stock in self._stocks:
            stock.reset()
        for idx in range(len(self._conf["stocks"])):
            self._stocks[idx].set_init_state(self._conf["stocks"][idx])

        # self._matrices_node.reset()

        self._pending_orders.clear()
        self._day_trade_orders.clear()
        self._finished_action.clear()

    def get_agent_idx_list(self) -> List[int]:
        """Get a list of stock index.

        Returns:
            list: List of stock index.
        """
        return [stock.code for stock in self._stocks]

    def get_metrics(self) -> dict:
        """Get current metrics information.

        Note:
            Call this method at different time will get different result.

        Returns:
            dict: Metrics information.
        """

        return DocableDict(
            METRICS_DESC,
            operation_number=len(self._finished_action)
        )

    def __del__(self):
        """Collect resource by order."""

        self._quote_pickers.clear()

        for reader in self._quote_readers:
            if reader:
                # Close binary reader first, so that we can clean it correctly.
                reader.close()

    def _load_configs(self):
        """Load configurations"""
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._conf = safe_load(fp)
            self._beginning_timestamp = datetime.datetime.strptime(self._conf["beginning_date"], "%Y-%m-%d").timestamp()

    def _init_frame(self):
        self._frame = build_frame(len(self._conf["stocks"]), self.calc_max_snapshots())
        self._snapshots = self._frame.snapshots
        self._account: Account = self._frame.account[0]
        self._account.set_init_state(init_money=self._conf['account']['money'])
        self._stocks = self._frame.stocks

        for idx in range(len(self._conf["stocks"])):
            self._stocks[idx].set_init_state(self._conf["stocks"][idx])

    def _register_events(self):
        # Register our own events and their callback handlers.
        self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_recieved)

    def is_market_closed(self):
        # TODO: Implement
        return True

    def _on_action_recieved(self, evt: Event):
        actions = evt.payload
        if actions is None:
            return

        for action in actions:
            if action is None:
                continue
            elif isinstance(action, CancelOrder):
                self._cancel_order(action=action, tick=evt.tick)
            else:  # Action is Order
                results: list = self._take_order(
                    order=action, tick=evt.tick
                )
                for result in results:
                    self._account.take_trade(
                        action, result
                    )

            if action.state != ActionState.pending and action.id not in self._finished_action:
                self._finished_action[action.id] = action
        # append panding orders still not triggered in the available life time
        if evt.tick in self._pending_orders:
            for action in self._pending_orders[evt.tick]:
                self._event_buffer.gen_atom_event(evt.tick, DecisionEvent, action)
            self._pending_orders[evt.tick].clear()

    def _take_order(self, order: Order, tick: int) -> TradeResult:
        """Deal with the order.
        """
        # 1. can trade -> bool
        # 2. return (stock, sell/busy, stock_price, number, tax)
        # 3. update stock.account_hold_num
        rets = []
        order_succeeded = False
        assert(order.state == ActionState.pending)
        # not canceled
        stock = self._stocks[order.item]
        is_trigger = order.is_trigger(stock.opening_price, stock.trade_volume)
        if is_trigger:
            print(f"  <<>>  executing {order}")
            if not self._conf["trade_constraint"]["allow_split"]:
                order_succeeded, rets = self._trade(action=order)
            else:
                order_succeeded, rets = self._split_trade(action=order)

            if order_succeeded:
                order.state = ActionState.success
                order.finish_tick = tick
                for ret in rets:
                    print(f"  <<>>  trading: {ret}")
                    if order.direction == OrderDirection.buy:
                        self._stocks[order.item].average_cost = (
                            self._stocks[order.item].account_hold_num * self._stocks[order.item].average_cost
                            + ret.price_per_item * ret.trade_number
                        ) / (self._stocks[order.item].account_hold_num + ret.trade_number)
                        # day trading is considered
                        if self._conf["trade_constraint"]["allow_day_trade"]:
                            self._stocks[order.item].account_hold_num += ret.trade_number
                        else:
                            if order not in self._day_trade_orders:
                                self._day_trade_orders.append(order)
                    else:
                        self._stocks[order.item].account_hold_num -= ret.trade_number
            else:
                order.state = ActionState.failed
                order.finish_tick = tick
                order.comment = "Can not execute the order"
        else:
            if order.remaining_life_time > 1:
                order.remaining_life_time -= 1
                # move to _on_action_recieved
                # self._event_buffer.gen_atom_event(tick + 1, DecisionEvent, action)
                if tick + 1 not in self._pending_orders:
                    self._pending_orders[tick + 1] = []
                self._pending_orders[tick + 1].append(order)
            else:
                order.state = ActionState.expired
                order.finish_tick = tick
        order.action_result = rets

        return rets

    def _cancel_order(self, action: CancelOrder, tick: int):
        """Cancel the specified order.
        """
        action.order.state = ActionState.canceled
        action.order.finish_tick = tick
        action.order.remaining_life_time = 0
        assert((tick in self._pending_orders) and (action.order in self._pending_orders[tick]))
        self._pending_orders[tick].remove(action.order)
        action.state = ActionState.success
        action.finish_tick = tick

    def _action_scope(self, action_type: ActionType, stock_index: int, tick: int):
        """Generate the action scope of the action
        """
        if action_type == ActionType.order:
            # action_scope of stock
            stock: Stock = self._stocks[stock_index]
            sell_max = 0
            sell_min = 0
            buy_max = 0
            buy_min = 0
            if stock.account_hold_num > 0:
                sell_max = stock.account_hold_num
                odd_shares = stock.account_hold_num % self._conf["trade_constraint"]["min_sell_unit"]
                if odd_shares > 0:
                    sell_min = odd_shares
                else:
                    sell_min = self._conf["trade_constraint"]["min_sell_unit"]

            if self._account.remaining_cash > stock.opening_price * self._conf["trade_constraint"]["min_buy_unit"]:
                buy_min = self._conf["trade_constraint"]["min_buy_unit"]
                buy_max = int(self._account.remaining_cash / stock.opening_price)
                buy_max = buy_max - buy_max % self._conf["trade_constraint"]["min_buy_unit"]

            if not self._conf["trade_constraint"]["allow_split"]:
                sell_max = min(sell_max, int(self._conf["trade_constraint"]["max_trade_percent"] * stock.trade_volume))
                sell_min = min(sell_min, sell_max)
                buy_max = min(buy_max, int(self._conf["trade_constraint"]["max_trade_percent"] * stock.trade_volume))
                buy_min = min(buy_min, buy_max)

            return OrderActionScope(buy_min, buy_max, sell_min, sell_max, self._support_order_mode)

        elif action_type == ActionType.cancel_order:
            # action_scope of pending orders
            available_orders = []
            if tick in self._pending_orders:
                available_orders = self._pending_orders[tick]
            return CancelOrderActionScope(available_orders)

    def _tick_2_date(self, tick: int):
        # Get current date to update additional info.
        # NOTE: We do not need hour and minutes for now.
        return (self._quote_start_date + relativedelta(minutes=tick)).date()

    def _split_trade(self, action: Order):
        rets = []
        trade_succeeded = False
        market_price = self._pick_market_price(self._stocks[action.item])
        market_volume = self._stocks[action.item].trade_volume

        slippage = ByMoneySlippage(self._conf["trade_constraint"]["slippage"])
        commission = ByMoneyCommission(self._conf["trade_constraint"]["commission"], 5)
        tax = StampTaxCommission(self._conf["trade_constraint"]["close_tax"])

        remaining_volume = action.amount
        max_trade_volume = self._conf["trade_constraint"]["max_trade_percent"] * market_volume
        if remaining_volume > max_trade_volume:
            remaining_volume = max_trade_volume
        if action.direction == OrderDirection.sell:
            remaining_volume = min(remaining_volume, self._stocks[action.item].account_hold_num)

        remaining_cash = self._account.remaining_cash

        split_volume = self._conf["trade_constraint"]["split_trade_percent"] * market_volume
        odd_shares = self._stocks[action.item].account_hold_num % self._conf["trade_constraint"]["min_sell_unit"]
        executing_price = market_price

        executing = True
        while executing:
            executing_volume = min(remaining_volume, split_volume)

            adjusted = True
            while adjusted:
                adjusted = False

                executing_price = slippage.execute(action.direction, executing_volume, executing_price, market_volume)

                executing_commission = commission.execute(action.direction, executing_price, executing_volume)

                executing_tax = tax.execute(action.direction, executing_price, executing_volume)

                # constraint
                if not self._validate_trade_on_volume_unit(
                    odd_shares=odd_shares, direction=action.direction, executing_price=executing_price,
                    executing_volume=executing_volume
                ):
                    executing_volume = self._adjust_trade_on_volume_unit(
                        odd_shares=odd_shares, direction=action.direction, executing_price=executing_price,
                        executing_volume=executing_volume
                    )
                    adjusted = True
                    odd_shares = 0
                    # print(f"Adjust volume for constraint:{executing_volume}")
                    continue

                # money
                if not self._validate_trade_on_remaining_cash(
                    direction=action.direction, executing_price=executing_price, executing_volume=executing_volume,
                    executing_commission=executing_commission, executing_tax=executing_tax,
                    remaining_cash=remaining_cash
                ):
                    executing_volume = self._adjust_trade_on_remaining_cash(
                        direction=action.direction, executing_price=executing_price, executing_volume=executing_volume,
                        executing_commission=executing_commission, executing_tax=executing_tax,
                        remaining_cash=remaining_cash
                    )
                    adjusted = True
                    # print(f"Adjust volume for trade:{executing_volume}")
                    continue

            if executing_volume > 0:
                ret = TradeResult(
                    direction=action.direction, trade_number=executing_volume,
                    price_per_item=executing_price, commission=executing_commission, tax=executing_tax
                )
                trade_succeeded = True
                remaining_volume -= executing_volume
                if action.direction == OrderDirection.buy:
                    remaining_cash -= ret.trade_number * ret.price_per_item + ret.tax
                else:
                    remaining_cash += ret.trade_number * ret.price_per_item - ret.tax

                if remaining_volume < 0 or remaining_cash < 0:
                    executing = False
                else:
                    rets.append(ret)
            else:
                executing = False

        return trade_succeeded, rets

    def _trade(self, action: Order):
        rets = []
        trade_succeeded = False
        market_price = self._pick_market_price(self._stocks[action.item])
        market_volume = self._stocks[action.item].trade_volume
        remaining_cash = self._account.remaining_cash

        slippage = ByMoneySlippage(self._conf["trade_constraint"]["slippage"])
        commission = ByMoneyCommission(self._conf["trade_constraint"]["commission"], 5)
        tax = StampTaxCommission(self._conf["trade_constraint"]["close_tax"])

        executing_volume = action.amount
        max_trade_volume = self._conf["trade_constraint"]["max_trade_percent"] * market_volume
        if executing_volume > max_trade_volume:
            executing_volume = max_trade_volume
        odd_shares = self._stocks[action.item].account_hold_num % self._conf["trade_constraint"]["min_sell_unit"]
        # print("action.amount", action.amount)

        if action.direction == OrderDirection.sell:
            executing_volume = min(executing_volume, self._stocks[action.item].account_hold_num)

        adjusted = True

        while adjusted:
            adjusted = False

            executing_price = market_price

            executing_price = slippage.execute(action.direction, executing_volume, executing_price, market_volume)

            executing_commission = commission.execute(action.direction, executing_price, executing_volume)

            executing_tax = tax.execute(action.direction, executing_price, executing_volume)

            # constraint
            if not self._validate_trade_on_volume_unit(
                odd_shares=odd_shares, direction=action.direction, executing_price=executing_price,
                executing_volume=executing_volume
            ):
                executing_volume = self._adjust_trade_on_volume_unit(
                    odd_shares=odd_shares, direction=action.direction, executing_price=executing_price,
                    executing_volume=executing_volume
                )
                adjusted = True
                # print(f"Adjust volume for constraint:{executing_volume}")
                continue

            # money
            if not self._validate_trade_on_remaining_cash(
                direction=action.direction, executing_price=executing_price, executing_volume=executing_volume,
                executing_commission=executing_commission, executing_tax=executing_tax,
                remaining_cash=remaining_cash
            ):
                executing_volume = self._adjust_trade_on_remaining_cash(
                    direction=action.direction, executing_price=executing_price, executing_volume=executing_volume,
                    executing_commission=executing_commission, executing_tax=executing_tax,
                    remaining_cash=remaining_cash
                )
                adjusted = True
                # print(f"Adjust volume for trade:{executing_volume}")
                continue

        if executing_volume > 0:
            ret = TradeResult(
                direction=action.direction, price_per_item=executing_price, trade_number=executing_volume,
                commission=executing_commission, tax=executing_tax
            )
            trade_succeeded = True
            rets.append(ret)

        return trade_succeeded, rets

    def _validate_trade_on_volume_unit(
        self, odd_shares: int, direction: OrderDirection, executing_price: float, executing_volume: int
    ):
        ret = True
        if direction == OrderDirection.buy:
            if executing_volume % self._conf["trade_constraint"]["min_buy_unit"] != 0:
                ret = False
        if direction == OrderDirection.sell:
            odd_volume = executing_volume % self._conf["trade_constraint"]["min_sell_unit"]
            if odd_volume != 0:
                if odd_shares != odd_volume:
                    ret = False

        return ret

    def _adjust_trade_on_volume_unit(
        self, odd_shares: int, direction: OrderDirection, executing_price: float, executing_volume: int
    ):
        ret = executing_volume
        if direction == OrderDirection.buy:
            odd_volume = ret % self._conf["trade_constraint"]["min_buy_unit"]
            ret -= odd_volume
        if direction == OrderDirection.sell:
            odd_volume = executing_volume % self._conf["trade_constraint"]["min_sell_unit"]
            if odd_volume != 0:
                ret -= odd_volume
                ret += odd_shares

        return int(ret)

    def _validate_trade_on_remaining_cash(
        self, direction: OrderDirection, executing_price: float,
        executing_volume: int, executing_commission: float, executing_tax: float, remaining_cash: float
    ):
        ret = True
        if direction == OrderDirection.buy:
            money_needed = executing_price * executing_volume + executing_commission + executing_tax
            if money_needed > remaining_cash:
                ret = False
        else:
            money_needed = executing_commission + executing_tax
            if money_needed > remaining_cash + executing_price * executing_volume:
                ret = False

        return ret

    def _adjust_trade_on_remaining_cash(
        self, direction: OrderDirection, executing_price: float,
        executing_volume: int, executing_commission: float, executing_tax: float, remaining_cash: float
    ):
        ret = executing_volume
        if direction == OrderDirection.buy:
            money_needed = executing_price * executing_volume + executing_commission + executing_tax
            ret = ret - math.ceil(
                (money_needed - remaining_cash) / (executing_price * self._conf["trade_constraint"]["min_buy_unit"])
            ) * self._conf["trade_constraint"]["min_buy_unit"]
        else:
            money_needed = executing_commission + executing_tax
            ret = ret - math.ceil(
                (money_needed - (remaining_cash + executing_price * ret)) / (
                    executing_price * self._conf["trade_constraint"]["min_buy_unit"]
                )
            ) * self._conf["trade_constraint"]["min_buy_unit"]
        return int(ret)

    def _pick_market_price(self, stock: Stock):
        return getattr(stock, self._conf["trade_constraint"]["deal_price"], stock.opening_price)
