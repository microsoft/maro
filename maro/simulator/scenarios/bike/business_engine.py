
import csv
import math
import os
import random
from enum import IntEnum
from math import floor, ceil
from typing import Dict, List

import holidays
from yaml import safe_load

from maro.simulator.event_buffer import DECISION_EVENT, Event, EventBuffer
from maro.simulator.frame import Frame, SnapshotList
from maro.simulator.scenarios import AbsBusinessEngine
from maro.simulator.utils.random import random

from .adj_reader import read_adj_info
from .cell import Cell
from .cell_reward import CellReward
from .common import (Action, BikeReturnPayload, BikeTransferPayload,
                     DecisionEvent, Trip, ExtraCostMode)
from .decision_strategy import BikeDecisionStrategy
from .frame_builder import build
from .trip_reader import BikeTripReader
from .weather_table import WeatherTable

bikes_adjust_rand = random["bikes_adjust"]


class BikeEventType(IntEnum):
    """Events we need to handled to process trip logic"""
    TripRequirement = 1    # a user need a bike
    BikeReturn = 2         # user return the bike at target cell
    BikeReceived = 3  # transfer bikes from a cell to another


class BikeBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, config_path: str, start_tick: int, max_tick: int, frame_resolution: int):
        super().__init__(event_buffer, config_path, start_tick, max_tick, frame_resolution)

        self._conf = None
        self._decision_strategy = None
        self._cells = []
        self._us_holidays = holidays.US()  # holidays for US, as we are using NY data

        self._reg_event()
        self._read_config()
        self._init_data_reader() # NOTE: we should read the data first, to get correct max tick if max_tick is -1
        self._init_frame()

        self._extra_cost_mode = ExtraCostMode(self._conf["extra_cost_mode"])
        frame_num = ceil(self._max_tick / frame_resolution)
        
        self._snapshots = SnapshotList(self._frame, frame_num)

        self._adj = read_adj_info(self._conf["adj_file"])
        self._decision_strategy = BikeDecisionStrategy(self._cells, self._conf["decision"])
        self._reward = CellReward(self._cells, self._conf["reward"])
        self._weather_table = WeatherTable(self._conf["weather_file"], self._data_reader.start_date)

        self._trip_adjust_rate = self._conf["trip_adjustment"]["adjust_rate"]
        self._trip_adjust_value = self._conf["trip_adjustment"]["adjust_value"]

        self._update_cell_adj()

    @property
    def frame(self) -> Frame:
        """Frame: Frame of current business engine
        """
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        """SnapshotList: Snapshot list of current frame"""
        return self._snapshots

    def step(self, tick: int):
        """Used to process events at specified tick, usually this is called by Env at each internal tick

        Args:
            tick (int): tick to process
        """

        # get trip requirements for current internal tick (in minute)
        trips = self._data_reader.get_trips(tick)

        # generate events to process
        for trip in trips:
            trip_evt = self._event_buffer.gen_atom_event(tick, BikeEventType.TripRequirement, payload=trip)

            self._event_buffer.insert_event(trip_evt)

        cells_need_decision = self._decision_strategy.get_cells_need_decision(tick)

        # the env will take snapshot for use when we need an action, so we do not need to take action here
        for cell_idx in cells_need_decision:
            decision_payload = DecisionEvent(cell_idx, tick, 
                     floor((tick - self._start_tick) / self._frame_resolution),
                     self._decision_strategy.action_scope)
            decision_evt = self._event_buffer.gen_cascade_event(tick, DECISION_EVENT, decision_payload)

            self._event_buffer.insert_event(decision_evt)

        return self._max_tick == tick + 1 # last tick

    @property
    def configs(self) -> dict:
        """object: Configurations of this business engine"""
        return self._conf

    def rewards(self, actions) -> float:
        """Calculate rewards based on actions

        Args:
            actions(list): Action(s) from agent

        Returns:
            float: reward based on actions
        """
        rewards = [self._reward.reward(action.from_cell) for action in actions]

        return rewards

    def reset(self):
        """Reset business engine"""
        # clear value in snapshots
        self._snapshots.reset()

        # clear value in current frame
        self._frame.reset()

        # prepare data pointer to beginning
        self._data_reader.reset()

        # reset cell to initial value
        for cell in self._cells:
            cell.reset()

    def get_node_name_mapping(self) -> Dict[str, Dict]:
        """Get node name mappings related with this environment

        Returns:
            Dict[str, Dict]: Node name to index mapping dictionary
        """
        return {}

    def get_agent_idx_list(self) -> List[int]:
        """Get port index list related with this environment

        Returns:
            List[int]: list of port index
        """
        return [i for i in range(len(self._cells))]

    def post_step(self, tick: int):
        """Post-process at specified tick

        Args:
            tick (int): tick to process

        """
        if (tick + 1) % self._frame_resolution == 0:
            # take a snapshot at the end of tick
            self._snapshots.insert_snapshot(self._frame, floor((tick - self._start_tick)/self._frame_resolution))

            # last unit tick of current tick
            # we will reset some field
            for cell in self._cells:
                cell.shortage = 0
                cell.trip_requirement = 0
                cell.unknow_gendors = 0
                cell.males = 0
                cell.females = 0
                cell.weekday = 0
                cell.customer = 0
                cell.subscriptor = 0
                cell.extra_cost = 0

    def _init_frame(self):
        rows = []
        with open(self._conf["cell_file"]) as fp:
            reader = csv.DictReader(fp)

            for l in reader:
                rows.append(l)

        self._frame = build(len(rows))

        bike_discount = 1

        if "bike_discount" in self._conf["trip_adjustment"]:
            bike_discount = float(
                self._conf["trip_adjustment"]["bike_discount"])

        for r in rows:
            if len(r) == 0:
                break

            cell = Cell(int(r["cell_id"]), int(r["capacity"]), round(
                int(r["init"]) * bike_discount), self._frame)

            self._cells.append(cell)

    def _init_data_reader(self):
        self._data_reader = BikeTripReader(self._conf["trip_file"],
                                           self._start_tick,
                                           self._max_tick)

        self._max_tick = self._data_reader.max_tick

    def _read_config(self):
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._conf = safe_load(fp)

    def _update_cell_adj(self):
        for cell in self._cells:
            cell.set_neighbors(self._adj[cell.index])

    def _reg_event(self):
        self._event_buffer.register_event_handler(BikeEventType.TripRequirement, self._on_trip_requirement)
        self._event_buffer.register_event_handler(BikeEventType.BikeReturn, self._on_bike_return)

        # decision event, predefined in event buffer
        self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_received)
        self._event_buffer.register_event_handler(BikeEventType.BikeReceived, self._on_bike_received)

    def _move_to_neighbor(self, src_cell: Cell, cell: Cell, bike_number: int, step: int = 1):
        cost = 0

        # move to 1-step neighbors
        for neighbor_idx in cell.neighbors:

            # ignore source cell and padding cell
            if neighbor_idx == src_cell.index or neighbor_idx < 0:
                continue

            neighbor = self._cells[neighbor_idx]
            accept_number = neighbor.capacity - neighbor.bikes

            # how many bikes this cell can accept
            accept_number = min(accept_number, bike_number)
            neighbor.bikes += accept_number
            cost += accept_number

            bike_number = bike_number - accept_number

            if bike_number == 0:
                break

        # cost of current step
        cost = self._calculate_extra_cost(cost, step)

        if step == 1 and bike_number > 0:
            # 2-step neighbors
            for neighbor_idx in cell.neighbors:
                if neighbor_idx < 0:
                    continue

                cost += self._move_to_neighbor(src_cell, self._cells[neighbor_idx], bike_number, step=2)

                if bike_number == 0:
                    break

            # if there still some more bikes, return it to source cell
            if bike_number > 0:
                src_cell.bikes += bike_number

                # TODO: remove hard coded step
                cost += self._calculate_extra_cost(bike_number, 3)

        return cost

    def _calculate_extra_cost(self, number: int, step: int = 1):
        return number * step

    def _on_trip_requirement(self, evt: Event):
        """On trip requirement handler:
        1. try to fulfill the trip requirement in this cell
        2. update inventory, shortage, gendor, usertype and weekday statistics states"""

        trip: Trip = evt.payload
        cell_idx: int = trip.from_cell
        cell: Cell = self._cells[cell_idx]
        cell_bikes = cell.bikes

        adjusted_number = trip.number

        # disable adjust if the rate is less equal 0
        if self._trip_adjust_rate > 0:
            adjusted_number += (self._trip_adjust_value if bikes_adjust_rand.random() < self._trip_adjust_rate else 0)

        # update trip count
        cell.trip_requirement += adjusted_number

        shortage = adjusted_number - cell_bikes
        executed_num = adjusted_number

        # update shortage and execute number if we have shortage
        if shortage > 0:
            cell.shortage += shortage

            executed_num = adjusted_number - shortage

        if executed_num > 0:
            cell.bikes = cell_bikes - executed_num

            cell.update_gendor(trip.gendor)
            cell.update_usertype(trip.usertype)

            # TODO: we can update following fields when the day is changed to save time
            weather = self._weather_table[trip.date]

            cell.weekday = trip.weekday
            cell.holiday = trip.date in self._us_holidays
            cell.weather = weather.type
            cell.temperature = weather.avg_temp

            # generate a bike return event by end tick
            return_payload = BikeReturnPayload(trip.from_cell, trip.to_cell, executed_num)
            bike_return_evt = self._event_buffer.gen_atom_event(trip.end_tick, BikeEventType.BikeReturn, payload=return_payload)

            self._event_buffer.insert_event(bike_return_evt)

    def _on_bike_return(self, evt: Event):
        payload: BikeReturnPayload = evt.payload
        cell: Cell = self._cells[payload.to_cell]

        cell_bikes = cell.bikes
        cell_capacity = cell.capacity
        return_number = payload.number

        if cell_bikes + return_number > cell_capacity:
            return_number = cell_capacity - cell_bikes

        if return_number > 0:
            cell.bikes += return_number

        if payload.number != return_number:
            # extra cost of current cell, as we do not know whose action caused this
            cell.extra_cost += self._move_to_neighbor(self._cells[payload.from_cell], cell, payload.number - return_number)

    def _on_action_received(self, evt: Event):
        action: Action = None

        for action in evt.payload:
            from_cell_idx: int = action.from_cell
            to_cell_idx: int = action.to_cell

            # ignore invalid cell idx
            if from_cell_idx < 0 or to_cell_idx < 0:
                continue

            cell: Cell = self._cells[from_cell_idx]

            executed_number = min(cell.bikes, action.number)

            # insert into event buffer if we have bikes to transfer
            if executed_number > 0:
                cell.bikes -= executed_number

                payload = BikeTransferPayload(from_cell_idx, to_cell_idx, executed_number)

                transfer_time = self._decision_strategy.transfer_time
                transfer_evt = self._event_buffer.gen_atom_event(evt.tick + transfer_time,
                                                                 BikeEventType.BikeReceived, payload)

                self._event_buffer.insert_event(transfer_evt)

    def _on_bike_received(self, evt: Event):
        payload: BikeTransferPaylod = evt.payload
        cell: Cell = self._cells[payload.to_cell]

        cell_bikes = cell.bikes
        cell_capacity = cell.capacity
        accept_number = payload.number

        if cell_bikes + accept_number > cell_capacity:
            accept_number = cell_capacity - cell_bikes
            extra_bikes = payload.number - accept_number

            extra_cost = self._move_to_neighbor(self._cells[payload.from_cell], cell, extra_bikes)

            if self._extra_cost_mode == ExtraCostMode.Source:
                # extra cost from source cell
                from_cell = self._cells[payload.from_cell]
                from_cell.extra_cost += extra_cost
            elif self._extra_cost_mode == ExtraCostMode.Target:
                cell.extra_cost += extra_cost
            elif self._extra_cost_mode == ExtraCostMode.TargetNeighbors:
                for neighbor_idx in cell.neighbors:
                    if neighbor_idx > 0:
                        neighbor: Cell = self._cells[neighbor_idx]

                        # TODO: shall we avg this value to neighbors?
                        neighbor.extra_cost += extra_cost

        cell.bikes += accept_number
