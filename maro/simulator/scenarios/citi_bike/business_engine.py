# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
import datetime
import numpy as np
import holidays

from csv import DictReader
from math import ceil, floor
from typing import List, Union

from yaml import safe_load
from dateutil.relativedelta import relativedelta
from dateutil.tz import UTC, gettz

from maro.backends.frame import FrameBase, SnapshotList
from maro.cli.data_pipeline.citi_bike import CitiBikeProcess
from maro.cli.data_pipeline.utils import chagne_file_path
from maro.data_lib import BinaryReader
from maro.event_buffer import DECISION_EVENT, Event, EventBuffer
from maro.simulator.scenarios import AbsBusinessEngine
from maro.simulator.scenarios.helpers import MatrixAttributeAccessor, DocableDict
from maro.utils.exception.cli_exception import CommandError
from maro.utils.logger import CliLogger

from .common import BikeReturnPayload, BikeTransferPayload, DecisionEvent, Action
from .decision_strategy import BikeDecisionStrategy
from .events import CitiBikeEvents
from .frame_builder import build_frame
from .station import Station
from .stations_info import StationInfo, get_station_info
from .weather_table import WeatherTable
from .adj_loader import load_adj_from_csv
from .station_reward import StationReward

logger = CliLogger(name=__name__)

metrics_desc = """
Citi bike metrics used to provide statistics information at current point (may be in the middle of a tick), it contains following keys:

trip_requirements (int): accumulative trips until now 

bike_shortage (int): accumulative shortage until now 

operation_number (int): accumulative operation cost until now 

"""

class CitibikeBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, topology: str, start_tick: int, max_tick: int, snapshot_resolution: int, max_snapshots:int, additional_options: dict = {}):
        super().__init__("citi_bike", event_buffer, topology, start_tick, max_tick, snapshot_resolution, max_snapshots, additional_options)

        # trip binary reader
        self._trip_reader: BinaryReader = None

        # update self._config_path with current file path
        self.update_config_root_path(__file__)

        # holidays for US, as we are using NY data
        self._us_holidays = holidays.US()  

        # our stations list used for quick accessing
        self._stations: List[Station] = []

        self._total_trips: int = 0
        self._total_shortages: int = 0
        self._total_operate_num: int = 0

        self._init()

    @property
    def frame(self) -> FrameBase:
        """Current frame"""
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        """Current snapshot list"""
        return self._snapshots

    @property
    def configs(self):
        return self._conf

    def rewards(self, actions) -> Union[float, list] :
        """Calculate rewards based on actions

        Args:
            actions(list): Action(s) from agent

        Returns:
            float: reward based on actions
        """
        if actions is None:
            return []

        return sum([self._reward.reward(station.index) for station in self._stations])

    def step(self, tick: int):
        """Push business engine to next step"""
        # if we do not set auto event, then we need to push it manually
        for trip in self._item_picker.items(tick):
            # generate a trip event, to dispatch to related callback to process this requirement
            trip_evt = self._event_buffer.gen_atom_event(tick, CitiBikeEvents.RequireBike, payload=trip)

            self._event_buffer.insert_event(trip_evt)

        if self._decision_strategy.is_decision_tick(tick):
            # generate an event, so that we can do the checking after all the trip requirement processed
            decition_checking_evt = self._event_buffer.gen_atom_event(tick, CitiBikeEvents.RebalanceBike)

            self._event_buffer.insert_event(decition_checking_evt)

        # update our additional features that not trip related
        self._update_station_extra_features(tick)

    def post_step(self, tick: int):
        # we following the snapshot_resolution settings to take snapshot
        if (tick + 1) % self._snapshot_resolution == 0:
            # NOTE: we should use frame_index method to get correct index in snapshot list
            self._frame.take_snapshot(self.frame_index(tick))

            # we reset the station station each resolution
            for station in self._stations:
                station.shortage = 0
                station.trip_requirement = 0
                station.extra_cost = 0
                station.transfer_cost = 0
                station.fulfillment = 0
                station.failed_return = 0
                station.min_bikes = station.bikes

        # stop current episode if we reach max tick
        return tick + 1 == self._max_tick

    def get_node_mapping(self)->dict:
        node_mapping = {}
        for station in self._stations:
            node_mapping[station.index] = station.id
        return node_mapping

    def reset(self):
        """Reset after episode"""
        self._total_trips = 0
        self._total_operate_num = 0
        self._total_shortages = 0

        self._frame.reset()

        self._snapshots.reset()

        self._trip_reader.reset()

        self._item_picker = self._trip_reader.items_tick_picker(self._start_tick, self._max_tick, time_unit="m")

        for station in self._stations:
            station.reset()

        self._matrices_node.reset()
        
        self._decision_strategy.reset()

    def get_agent_idx_list(self) -> List[int]:
        return [station.index for station in self._stations]

    def get_metrics(self) -> dict:
        """metrics information"""
        total_trips = self._total_trips
        total_shortage = self._total_shortages

        return DocableDict(metrics_desc, 
                trip_requirements = total_trips, 
                bike_shortage = total_shortage,
                operation_number = self._total_operate_num)

    def __del__(self):
        """Collect resource by order"""

        self._item_picker = None

        if self._trip_reader:
            # close binary reader first, so that we can clean it correctly
            self._trip_reader.close()


    def _init(self):
        self._load_configs()
        self._register_events()
        self._citi_bike_data_pipeline = None

        # time zone we used to transfer UTC to target time zone
        self._time_zone = gettz(self._conf["time_zone"])

        # our weather table used to query weather by date
        weather_data_path = self._conf["weather_data"]
        if weather_data_path.startswith("~"):
            weather_data_path = os.path.expanduser(weather_data_path) 
        
        trip_data_path = self._conf["trip_data"]
        if trip_data_path.startswith("~"):
            trip_data_path = os.path.expanduser(trip_data_path)

        # TODO: Weather data source changed, temporarily disable, will enable it later when new data source is available.
        # if (not os.path.exists(weather_data_path)) or (not os.path.exists(trip_data_path)):
        if not os.path.exists(trip_data_path):
            self._build_temp_data()

        # TODO: Weather data source changed, temporarily disable, will enable it later when new data source is available.
        # self._weather_lut = WeatherTable(self._conf["weather_data"], self._time_zone)

        self._trip_reader = BinaryReader(self._conf["trip_data"])

        # we keep this used to calculate real datetime to get weather and holiday info
        self._trip_start_date: datetime.datetime = self._trip_reader.start_datetime

        # since binary data hold UTC timestamp, convert it into our target timezone
        self._trip_start_date = self._trip_start_date.astimezone(self._time_zone)

        # used to cache last date we updated the station additional features to avoid to much time updating
        self._last_date: datetime.datetime = None

        # filter data with tick range by minute (time_unit='m')
        self._item_picker = self._trip_reader.items_tick_picker(self._start_tick, self._max_tick, time_unit="m")

        # we use this to init frame and stations init states
        stations_states = get_station_info(self._conf["stations_init_data"])

        self._init_frame(len(stations_states))

        self._init_stations(stations_states)
        
        self._init_adj_matrix()

        # our decision strategy to determine when we need an action
        self._decision_strategy = BikeDecisionStrategy(
            self._stations, self._distance_adj, self._snapshots, self._conf["decision"])

        self._reward = StationReward(self._stations, self._conf["reward"])

    def _load_configs(self):
        """Load configurations"""
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._conf = safe_load(fp)

    def _init_stations(self, stations_states: list):
        # after frame initializing, it will help us create the station instances, let's create a reference
        # the attribute is added by frame that as same defined in frame definition

        # NOTE: this is the build in station list that index start from 0,
        # we need to create a mapping for it, as our trip data only contains id
        self._stations = self._frame.stations

        for state in stations_states:
            # get related station, and set the init states
            station = self._stations[state.index]

            station.set_init_state(state.bikes, state.capacity, state.id)

    def _init_adj_matrix(self):
        # our distance adj
        # we assuming that our adj is NxN without header
        distance_adj = np.array(load_adj_from_csv(self._conf["distance_adj_data"], skiprows=1))
        
        # we only have one node here
        self._matrices_node = self._frame.matrices[0]
         
        station_num = len(self._stations)

        self._distance_adj = distance_adj.reshape(station_num, station_num)

        # add wrapper to it to make it easy to use,
        # with this we can get value by:
        # 1. self._trips_adj[x, y]
        # 2. self._trips_adj.get_row(0)
        # 3. self._trips_adj.get_column(0)
        self._trips_adj = MatrixAttributeAccessor(self._matrices_node, "trips_adj", station_num, station_num)

    def _init_frame(self, station_num: int):
        self._frame = build_frame(station_num, self.calc_max_snapshots())
        self._snapshots = self._frame.snapshots

    def _register_events(self):
        # register our own events and their callback handlers
        self._event_buffer.register_event_handler(CitiBikeEvents.RequireBike, self._on_required_bike)
        self._event_buffer.register_event_handler(CitiBikeEvents.ReturnBike, self._on_bike_returned)
        self._event_buffer.register_event_handler(CitiBikeEvents.RebalanceBike, self._on_rebalance_bikes)
        self._event_buffer.register_event_handler(CitiBikeEvents.DeliverBike, self._on_bike_deliver)

        # decision event, predefined in event buffer
        self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_received)

    def _tick_2_date(self, tick: int):
        # get current date to update additional info
        # NOTE: we do not need hour and minutes for now
        return (self._trip_start_date + relativedelta(minutes=tick)).date()

    def _update_station_extra_features(self, tick: int):
        """update features that not related to trips"""
        cur_datetime = self._tick_2_date(tick)

        if self._last_date == cur_datetime:
            return

        self._last_date = cur_datetime

        # TODO: Weather data source changed, temporarily disable, will enable it later when new data source is available.
        # weather_info = self._weather_lut[cur_datetime]

        weekday = cur_datetime.weekday()
        holiday = cur_datetime in self._us_holidays

        # default weather and temperature
        weather = 0
        temperature = 0

        # TODO: Weather data source changed, temporarily disable, will enable it later when new data source is available.
        # if weather_info is not None:
        #     weather = weather_info.weather
        #     temperature = weather_info.temp

        for station in self._stations:
            station.weekday = weekday
            station.holiday = holiday
            station.weather = weather
            station.temperature = temperature

    def _on_required_bike(self, evt: Event):
        """callback when there is a trip requirement generated"""

        trip = evt.payload
        station_idx: int = trip.src_station
        station: Station = self._stations[station_idx]
        station_bikes = station.bikes

        # update trip count, each item only contains 1 requirement
        station.trip_requirement += 1

        # statistics for metrics
        self._total_trips += 1

        self._trips_adj[station_idx, trip.dest_station] += 1

        if station_bikes < 1:
            station.shortage += 1
            self._total_shortages += 1
        else:
            station.fulfillment += 1
            station.bikes = station_bikes - 1

            # generate a bike return event by end tick
            return_payload = BikeReturnPayload(station_idx, trip.dest_station, 1)

            # durations from csv file is in seconds, convert it into minutes
            return_tick = evt.tick + trip.durations

            bike_return_evt = self._event_buffer.gen_atom_event(return_tick, CitiBikeEvents.ReturnBike, payload=return_payload)

            self._event_buffer.insert_event(bike_return_evt)

    def _on_bike_returned(self, evt: Event):
        """callback when there is a bike returned to a station"""
        payload: BikeReturnPayload = evt.payload

        station: Station = self._stations[payload.to_station_idx]

        station_bikes = station.bikes
        return_number = payload.number

        empty_docks = station.capacity - station_bikes

        max_accept_number = min(empty_docks, return_number)

        if max_accept_number < return_number:
            src_station = self._stations[payload.from_station_idx]

            additional_bikes = return_number - max_accept_number

            station.failed_return += additional_bikes

            # we have to move additional bikes to neighbors
            self._decision_strategy.move_to_neighbor(src_station, station, additional_bikes)

        station.bikes = station_bikes + max_accept_number

    def _on_rebalance_bikes(self, evt: Event):
        """callback when need to check if we should send decision event to agent"""

        # get stations that need an action
        stations_need_decision = self._decision_strategy.get_stations_need_decision(evt.tick)

        if len(stations_need_decision) > 0:
            # generate a decision event
            for station_idx, decision_type in stations_need_decision:
                decision_payload = DecisionEvent(station_idx, evt.tick,
                                                 self.frame_index(evt.tick),
                                                 self._decision_strategy.action_scope, decision_type)

                decision_evt = self._event_buffer.gen_cascade_event(evt.tick, DECISION_EVENT, decision_payload)

                self._event_buffer.insert_event(decision_evt)

    def _on_bike_deliver(self, evt: Event):
        """callback when our transferred bikes reach the destination"""
        payload: BikeTransferPayload = evt.payload
        station: Station = self._stations[payload.to_station_idx]

        station_bikes = station.bikes
        transfered_number = payload.number

        empty_docks = station.capacity - station_bikes

        max_accept_number = min(empty_docks, transfered_number)

        if max_accept_number < transfered_number:
            src_station = self._stations[payload.from_station_idx]

            self._decision_strategy.move_to_neighbor(src_station, station, transfered_number - max_accept_number)

        if max_accept_number > 0:
            station.transfer_cost += max_accept_number
            self._total_operate_num += max_accept_number

        station.bikes = station_bikes + max_accept_number


    def _on_action_received(self, evt: Event):
        """callback when we get an action from agent"""
        action: Action = None

        if evt is None or evt.payload is None:
            return

        for action in evt.payload:
            from_station_idx: int = action.from_station_idx
            to_station_idx: int = action.to_station_idx

            # ignore invalid cell idx
            if from_station_idx < 0 or to_station_idx < 0:
                continue

            station: Station = self._stations[from_station_idx]
            station_bikes = station.bikes

            executed_number = min(station_bikes, action.number)

            # insert into event buffer if we have bikes to transfer
            if executed_number > 0:
                station.bikes = station_bikes - executed_number

                payload = BikeTransferPayload(from_station_idx, to_station_idx, executed_number)

                transfer_time = self._decision_strategy.transfer_time
                transfer_evt = self._event_buffer.gen_atom_event(evt.tick + transfer_time,
                                                                 CitiBikeEvents.DeliverBike, payload)

                self._event_buffer.insert_event(transfer_evt)

    def _build_temp_data(self):
        """build temporary data for predefined environment"""
        logger.warning_yellow(f"Binary data files for scenario: citi_bike topology: {self._topology} not found.")
        citi_bike_process = CitiBikeProcess(is_temp=True)
        if self._topology in citi_bike_process.topologies:
            pid = str(os.getpid())
            logger.warning_yellow(
                f"Generating temp binary data file for scenario: citi_bike topology: {self._topology} pid: {pid}. If you want to keep the data, please use MARO CLI command 'maro env data generate -s citi_bike -t {self._topology}' to generate the binary data files first.")
            self._citi_bike_data_pipeline = citi_bike_process.topologies[self._topology]
            self._citi_bike_data_pipeline.download()
            self._citi_bike_data_pipeline.clean()
            self._citi_bike_data_pipeline.build()
            build_folders = self._citi_bike_data_pipeline.get_build_folders()
            trip_folder = build_folders["trip"]
            # TODO: Weather data source changed, temporarily disable, will enable it later when new data source is available.
            # weather_folder = build_folders["weather"]
            # self._conf["weather_data"] = chagne_file_path(self._conf["weather_data"], weather_folder)
            self._conf["trip_data"] = chagne_file_path(self._conf["trip_data"], trip_folder)
            self._conf["stations_init_data"] = chagne_file_path(self._conf["stations_init_data"], trip_folder)
            self._conf["distance_adj_data"] = chagne_file_path(self._conf["distance_adj_data"], trip_folder)
        else:
            raise CommandError("generate", f"Can not generate data files for scenario: citi_bike topology: {self._topology}")
