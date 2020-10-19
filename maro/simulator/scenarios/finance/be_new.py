# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import datetime
import os
from typing import List

import holidays
import numpy as np
from dateutil.relativedelta import relativedelta
from dateutil.tz import gettz
from maro.backends.frame import FrameBase, SnapshotList
from maro.cli.data_pipeline.citi_bike import CitiBikeProcess
from maro.cli.data_pipeline.utils import chagne_file_path
from maro.data_lib import BinaryReader
from maro.event_buffer import DECISION_EVENT, Event, EventBuffer
from maro.simulator.scenarios import AbsBusinessEngine
from maro.simulator.scenarios.helpers import DocableDict, MatrixAttributeAccessor
from maro.utils.exception.cli_exception import CommandError
from maro.utils.logger import CliLogger
from yaml import safe_load

from .adj_loader import load_adj_from_csv
from .common import Action, BikeReturnPayload, BikeTransferPayload, DecisionEvent
from .decision_strategy import BikeDecisionStrategy
from .events import CitiBikeEvents
from .frame_builder import build_frame
from .stock import Stock
from .stations_info import get_station_info
from .weather_table import WeatherTable

logger = CliLogger(name=__name__)

metrics_desc = """
Citi bike metrics used to provide statistics information at current point (may be in the middle of a tick).
It contains following keys:

trip_requirements (int): Accumulative trips until now.
bike_shortage (int): Accumulative shortage until now.
operation_number (int): Accumulative operation cost until now.
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

        self._init()

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
        # append account event
        decision_event = DecisionEvent(
            tick, FinanceType.stock, -1, self._action_scope, action_type=ActionType.transfer
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
            self._quote_pickers[idx]=self._quote_readers[idx].items_tick_picker(self._start_tick, self._max_tick, time_unit="d")

        for stock in self._stocks:
            stock.reset()

        self._matrices_node.reset()

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

    def _init(self):
        self._load_configs()
        self._register_events()
        self._finance_data_pipeline = None

        # Time zone we used to transfer UTC to target time zone.
        self._time_zone = gettz(self._conf["time_zone"])

        # Our weather table used to query weather by date.
        quote_data_path = self._conf["data_path"]
        if quote_data_path.startswith("~"):
            quote_data_path = os.path.expanduser(quote_data_path)

        quote_data_paths = []
        for idx in range(len(self._conf["stocks"])):
            quote_data_paths[idx] = os.path.join(quote_data_path, f"{self._conf['stocks'][idx]}.bin")

        all_quote_exists = True
        for data_path in quote_data_paths:
            if not os.path.exists(data_path):
                all_quote_exists = False
                break
        if not all_quote_exists:
            self._build_temp_data()

        for idx in range(len(self._conf["stocks"])):
            self._quote_readers[idx] = BinaryReader(quote_data_paths[idx])
            self._quote_pickers[idx] = self._quote_readers[idx].items_tick_picker(self._start_tick, self._max_tick, time_unit="d")

        # We keep this used to calculate real datetime to get weather and holiday info.
        self._trip_start_date: datetime.datetime = self._trip_reader.start_datetime

        # Since binary data hold UTC timestamp, convert it into our target timezone.
        self._trip_start_date = self._trip_start_date.astimezone(self._time_zone)

        # Used to cache last date we updated the station additional features to avoid to much time updating.
        self._last_date: datetime.datetime = None

        # We use this to initializing frame and stations states.
        stations_states = get_station_info(self._conf["stations_init_data"])

        self._init_frame(len(stations_states))

        self._init_stocks(self._conf["stocks"])

        self._init_adj_matrix()

    def _load_configs(self):
        """Load configurations"""
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._conf = safe_load(fp)
            self._beginning_timestamp = datetime.datetime.strptime(self._conf["beginning_date"], "%Y-%m-%d").timestamp()

    def _init_stocks(self, stock_codes: list):
        # After frame initializing, it will help us create the station instances, let's create a reference.
        # The attribute is added by frame that as same defined in frame definition.

        # NOTE: Tthis is the build in station list that index start from 0,
        # we need to create a mapping for it, as our trip data only contains id.
        self._stocks = self._frame.stocks

        for idx in range(len(stock_codes)):
            # Get related station, and set the init states.
            stock = self._stocks[idx]

            stock.set_init_state(stock_codes[idx])

    def _init_adj_matrix(self):
        # # Our distance adj. Assume that the adj is NxN without header.
        # distance_adj = np.array(load_adj_from_csv(self._conf["distance_adj_data"], skiprows=1))

        # We only have one node here.
        self._matrices_node = self._frame.matrices[0]

        stock_num = len(self._stocks)

        # self._distance_adj = distance_adj.reshape(station_num, station_num)

        # Add wrapper to it to make it easy to use, with this we can get value by:
        # 1. self._trips_adj[x, y].
        # 2. self._trips_adj.get_row(0).
        # 3. self._trips_adj.get_column(0).
        # self._trips_adj = MatrixAttributeAccessor(self._matrices_node, "trips_adj", station_num, station_num)

    def _init_frame(self, stock_num: int):
        self._frame = build_frame(stock_num, self.calc_max_snapshots())
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
                engine_name = action.sub_engine_name

                if engine_name in self._sub_engines:
                    if action.action_type == ActionType.transfer:
                        self._account.take_action(action, evt.tick)
                    elif action.action_type == ActionType.cancel_order:
                        self._sub_engines[engine_name].cancel_order(action)
                    elif action.action_type == ActionType.order:
                        result: TradeResult = self._sub_engines[engine_name].take_action(
                            action, self._account._sub_account_dict[engine_name].remaining_money, evt.tick
                        )
                        self._account.take_trade(
                            result, cur_data=self._sub_engines[engine_name]._stock_list, cur_engine=engine_name
                        )
                else:
                    raise "Specified engine not exist."
                if action.state != ActionState.pending and action.id not in self._finished_action:
                    self._finished_action[action.id] = action

    def _tick_2_date(self, tick: int):
        # Get current date to update additional info.
        # NOTE: We do not need hour and minutes for now.
        return (self._trip_start_date + relativedelta(minutes=tick)).date()

    def _build_temp_data(self):
        """Build temporary data for predefined environment."""
        logger.warning_yellow(f"Binary data files for scenario: citi_bike topology: {self._topology} not found.")
        citi_bike_process = CitiBikeProcess(is_temp=True)
        if self._topology in citi_bike_process.topologies:
            pid = str(os.getpid())
            logger.warning_yellow(
                f"Generating temp binary data file for scenario: citi_bike topology: {self._topology} pid: {pid}. "
                "If you want to keep the data, please use MARO CLI command "
                f"'maro env data generate -s citi_bike -t {self._topology}' to generate the binary data files first."
            )
            self._citi_bike_data_pipeline = citi_bike_process.topologies[self._topology]
            self._citi_bike_data_pipeline.download()
            self._citi_bike_data_pipeline.clean()
            self._citi_bike_data_pipeline.build()
            build_folders = self._citi_bike_data_pipeline.get_build_folders()
            trip_folder = build_folders["trip"]
            weather_folder = build_folders["weather"]
            self._conf["weather_data"] = chagne_file_path(self._conf["weather_data"], weather_folder)
            self._conf["trip_data"] = chagne_file_path(self._conf["trip_data"], trip_folder)
            self._conf["stations_init_data"] = chagne_file_path(self._conf["stations_init_data"], trip_folder)
            self._conf["distance_adj_data"] = chagne_file_path(self._conf["distance_adj_data"], trip_folder)
        else:
            raise CommandError(
                "generate", f"Can not generate data files for scenario: citi_bike topology: {self._topology}"
            )
