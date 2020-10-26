# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import datetime
import os
from typing import List
from collections import OrderedDict

from dateutil.relativedelta import relativedelta
from dateutil.tz import gettz
from maro.backends.frame import FrameBase, SnapshotList
from maro.data_lib import BinaryReader
from maro.event_buffer import DECISION_EVENT, Event, EventBuffer
from maro.simulator.scenarios import AbsBusinessEngine
from maro.simulator.scenarios.helpers import DocableDict
from maro.utils.logger import CliLogger
from yaml import safe_load

from .frame_builder import build_frame
from maro.simulator.scenarios.finance.stock.stock import Stock
from maro.simulator.scenarios.finance.common.common import (
    Action, DecisionEvent,
    FinanceType, TradeResult, OrderMode, ActionType, ActionState
)
from maro.simulator.scenarios.finance.stock.stock_trader import StockTrader

logger = CliLogger(name=__name__)

metrics_desc = """
"""


class FinanceBusinessEngine(AbsBusinessEngine):
    def __init__(
        self, event_buffer: EventBuffer, topology: str, start_tick: int,
        max_tick: int, snapshot_resolution: int, max_snapshots: int, additional_options: dict = {}
    ):
        super().__init__(
            "finance", event_buffer, topology, start_tick, max_tick,
            snapshot_resolution, max_snapshots, additional_options
        )

        # Trip binary reader.
        self._quote_readers: list = []
        self._quote_pickers: list = []

        # Update self._config_path with current file path.
        self.update_config_root_path(__file__)

        # Our stations list used for quick accessing.
        self._stocks: List[Stock] = []

        self._total_order_num: int = 0

        self._load_configs()
        self._register_events()
        self._finance_data_pipeline = None

        # Time zone we used to transfer UTC to target time zone.
        self._time_zone = gettz(self._conf["time_zone"])

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
            self._quote_pickers.append(
                self._quote_readers[idx].items_tick_picker(self._start_tick, self._max_tick, time_unit="d"))

        # We keep this used to calculate real datetime to get quote info.
        self._quote_start_date: datetime.datetime = datetime.datetime.fromisoformat(self._conf["beginning_date"])

        # Since binary data hold UTC timestamp, convert it into our target timezone.
        self._quote_start_date = self._quote_start_date.astimezone(self._time_zone)

        # Used to cache last date we updated the station additional features to avoid to much time updating.
        self._last_date: datetime.datetime = None

        self._init_frame()

        self._init_stocks()

        self._init_adj_matrix()

        self._order_mode = OrderMode.market_order
        self._trader = None

        self._action_scope_max = self._conf["action_scope"]["max"]

        self._init_trader(self._conf)

        self._pending_orders = []
        self._canceled_orders = []
        self._finished_action = OrderedDict()

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
                    stock: Stock = self._stocks[idx]
                    stock.fill(raw_stock)
                    if stock.is_valid:
                        valid_stocks.append(idx)

        # append cancel_order event
        decision_event = DecisionEvent(
            tick, FinanceType.stock, -2, self._action_scope, action_type=ActionType.cancel_order
        )
        evt = self._event_buffer.gen_cascade_event(tick, DecisionEvent, decision_event)
        self._event_buffer.insert_event(evt)

        # append order event
        for valid_stock in valid_stocks:
            decision_event = DecisionEvent(
                tick, FinanceType.stock, valid_stock,
                self._action_scope, action_type=ActionType.order
            )
            evt = self._event_buffer.gen_cascade_event(tick, DecisionEvent, decision_event)

            self._event_buffer.insert_event(evt)

    def post_step(self, tick: int):
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
        self._total_order_num = 0

        self._frame.reset()

        self._snapshots.reset()

        for reader in self._quote_readers:
            reader.reset()

        for idx in range(len(self._quote_pickers)):
            self._quote_pickers[idx] = self._quote_readers[idx].items_tick_picker(
                self._start_tick, self._max_tick, time_unit="d")

        for stock in self._stocks:
            stock.reset()

        # self._matrices_node.reset()

        self._pending_orders = []
        self._canceled_orders = []
        self._finished_action = OrderedDict()

    def get_agent_idx_list(self) -> List[int]:
        """Get a list of agent index.

        Returns:
            list: List of agent index.
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
            metrics_desc,
            operation_number=self._total_order_num
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

    def _init_stocks(self):
        # After frame initializing, it will help us create the station instances, let's create a reference.
        # The attribute is added by frame that as same defined in frame definition.

        # NOTE: Tthis is the build in station list that index start from 0,
        # we need to create a mapping for it, as our trip data only contains id.
        self._stocks = self._frame.stocks

        for idx in range(len(self._conf["stocks"])):
            # Get related station, and set the init states.
            stock = self._stocks[idx]
            stock.set_init_state(self._conf["stocks"][idx])

    def _init_adj_matrix(self):
        # # Our distance adj. Assume that the adj is NxN without header.
        # distance_adj = np.array(load_adj_from_csv(self._conf["distance_adj_data"], skiprows=1))

        # We only have one node here.
        # self._matrices_node = self._frame.matrices[0]

        # stock_num = len(self._stocks)

        # self._distance_adj = distance_adj.reshape(station_num, station_num)

        # Add wrapper to it to make it easy to use, with this we can get value by:
        # 1. self._trips_adj[x, y].
        # 2. self._trips_adj.get_row(0).
        # 3. self._trips_adj.get_column(0).
        # self._trips_adj = MatrixAttributeAccessor(self._matrices_node, "trips_adj", station_num, station_num)
        pass

    def _init_frame(self):
        self._frame = build_frame(len(self._conf["stocks"]), self.calc_max_snapshots())
        self._snapshots = self._frame.snapshots
        self._account = self._frame.account[0]
        self._account.set_init_state(
            init_money=self._conf['account']['money']
        )

    def _register_events(self):
        # Register our own events and their callback handlers.
        self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_recieved)

    def _on_action_recieved(self, evt: Event):
        actions = evt.payload
        if actions is None:
            return

        for action in actions:
            if action is None:
                pass
            else:
                if action.id not in self._account.action_history:
                    self._account.action_history[action.id] = action

                if action.action_type == ActionType.cancel_order:
                    self.cancel_order(action)
                elif action.action_type == ActionType.order:
                    result: TradeResult = self.take_action(
                        action, self._account.remaining_money, evt.tick
                    )
                    self._account.take_trade(
                        result, cur_data=self._stocks
                    )

                if action.state != ActionState.pending and action.id not in self._finished_action:
                    self._finished_action[action.id] = action

    def take_action(self, action: Action, remaining_money: float, tick: int) -> TradeResult:
        # 1. can trade -> bool
        # 2. return (stock, sell/busy, stock_price, number, tax)
        # 3. update stock.account_hold_num
        ret = TradeResult(action.item_index, 0, tick, 0, 0, False, False)
        if action.id not in self._canceled_orders:
            # not canceled
            out_of_scope, allow_split = self._verify_action(action)
            if not out_of_scope:
                if not allow_split:
                    asset, is_success, actual_price, actual_volume, commission_charge,\
                        is_trigger = self._trader.trade(
                            action, self._stocks, remaining_money
                        )  # list  index is in action # self.snapshot

                elif allow_split:
                    asset, is_success, actual_price, actual_volume, commission_charge, \
                        is_trigger = self._trader.split_trade(
                            action, self._stocks, remaining_money,
                            self._stocks[action.item_index].trade_volume * self._action_scope_max
                        )  # list  index is in action # self.snapshot
                ret = TradeResult(
                    action.item_index, actual_volume, tick, actual_price,
                    commission_charge, is_success, is_trigger
                )
                if not is_trigger:
                    if action.life_time != 1:
                        action.life_time -= 1
                        self._event_buffer.gen_atom_event(tick + 1, DecisionEvent, action)
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
                            self._stocks[asset].average_cost = (
                                (self._stocks[asset].account_hold_num * self._stocks[asset].average_cost) +
                                (actual_price * actual_volume)
                            ) / (self._stocks[asset].account_hold_num + actual_volume)
                        self._stocks[asset].account_hold_num += actual_volume
                    else:
                        action.state = ActionState.failed
                    action.finish_tick = tick
                    action.action_result = ret
            else:
                print(
                    "Warning: out of action scope and not allow split!",
                    self._action_scope(action.action_type, action.item_index), action.number
                )
                action.state = ActionState.failed
                action.finish_tick = tick

        else:
            # order canceled
            self._canceled_orders.remove(action.id)
            action.state = ActionState.canceled
            action.finish_tick = tick
        return ret

    def cancel_order(self, action: Action):
        if action.id in self._pending_orders:
            self._pending_orders.remove(action.id)
        if action.id not in self._canceled_orders:
            print(f'Order canceled :{action.id}')
            self._canceled_orders.append(action.id)

    def _action_scope(self, action_type: ActionType, stock_index: int):
        if action_type == ActionType.order:
            # action_scope of stock
            stock: Stock = self._stocks[stock_index]
            if self._allow_split:
                result = (-stock.account_hold_num, stock.trade_volume)
            else:
                result = (
                    max([-stock.account_hold_num, -stock.trade_volume * self._action_scope_max]),
                    stock.trade_volume * self._action_scope_max
                )
            return (result, self._trader.supported_orders, self._order_mode)

        elif action_type == ActionType.cancel_order:
            # action_scope of pending orders
            result = self._pending_orders

            return result

    def _init_trader(self, config):
        trade_constrain = self._conf['trade_constrain']

        self._trader = StockTrader(trade_constrain)

    def _verify_action(self, action: Action):
        ret = True
        allow_split = self._allow_split
        if self._action_scope(action.action_type, action.item_index)[0][0] <= action.number \
                and self._action_scope(action.action_type, action.item_index)[0][1] >= action.number:
            ret = False
        return ret, allow_split

    @property
    def _allow_split(self):
        allow_split = False
        if "allow_split" in self._conf["trade_constrain"]:
            allow_split = self._conf["trade_constrain"]["allow_split"]
        return allow_split

    def _tick_2_date(self, tick: int):
        # Get current date to update additional info.
        # NOTE: We do not need hour and minutes for now.
        return (self._quote_start_date + relativedelta(minutes=tick)).date()

    def _build_temp_data(self):
        """Build temporary data for predefined environment."""
        pass
