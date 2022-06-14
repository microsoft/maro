# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import io
import math
from typing import List, Tuple

import numpy as np
import yaml
from citi_bike_ilp import CitiBikeILP

from maro.data_lib import BinaryReader, ItemTickPicker
from maro.event_buffer import AbsEvent
from maro.forecasting import OneStepFixWindowMA as Forecaster
from maro.simulator import Env
from maro.simulator.scenarios.citi_bike.adj_loader import load_adj_from_csv
from maro.simulator.scenarios.citi_bike.common import Action, BikeReturnPayload, DecisionEvent
from maro.simulator.scenarios.citi_bike.events import CitiBikeEvents
from maro.utils import convert_dottable

# For debug only.
PEEP_AND_USE_REAL_DATA: bool = False
ENV: Env = None
TRIP_PICKER: ItemTickPicker = None


class MaIlpAgent:
    def __init__(
        self,
        ilp: CitiBikeILP,
        num_station: int,
        num_time_interval: int,
        ticks_per_interval: int,
        ma_window_size: int,
    ):
        """An agent that make decisions by ILP in Citi Bike scenario.

        Args:
            ilp (CitiBikeILP): The ILP instance.
            num_station (int): The number of stations in the target environment.
            num_time_interval (int): The number of time intervals for which the agent need to provide future demand and
                supply for the ILP. Also, the time interval in the agent indicates the number of environment ticks
                between two decision points in the ILP.
            ticks_per_interval (int): How many environment ticks in each time interval. It is same to the number of
                ticks between two decision points in the ILP.
            ma_window_size (int): The historical data maintain window size of the Moving Average Forecaster.
        """
        self._ilp = ilp
        self._num_station = num_station
        self._num_time_interval = num_time_interval
        self._ticks_per_interval = ticks_per_interval
        if not PEEP_AND_USE_REAL_DATA:
            self._demand_forecaster = [Forecaster(window_size=ma_window_size) for _ in range(self._num_station)]
            self._supply_forecaster = [Forecaster(window_size=ma_window_size) for _ in range(self._num_station)]
            self._num_recorded_interval = 0
            self._next_event_idx = 0

    # ============================= private start =============================

    def _record_history(self, env_tick: int, finished_events: List[AbsEvent]):
        """
        Args:
            env_tick (int): The current Env tick.
            finished_events (List[Event]): The finished events got from the Env.
        """
        num_interval_to_record = (env_tick - 1) // self._ticks_per_interval - self._num_recorded_interval
        if num_interval_to_record <= 0:
            return
        demand_history = np.zeros((num_interval_to_record, self._num_station), dtype=np.int16)
        supply_history = np.zeros((num_interval_to_record, self._num_station), dtype=np.int16)

        while self._next_event_idx < len(finished_events):
            # Calculate the interval index of this finished event.
            interval_idx = (
                finished_events[self._next_event_idx].tick // self._ticks_per_interval - self._num_recorded_interval
            )
            assert interval_idx >= 0, "The finished events are not sorted by tick!"
            if interval_idx >= num_interval_to_record:
                break

            # Check the event_type to get the historical demand and supply.
            event_type = finished_events[self._next_event_idx].event_type
            if event_type == CitiBikeEvents.RequireBike:
                # TODO: Replace it with a pre-defined PayLoad.
                payload = finished_events[self._next_event_idx].body
                demand_history[interval_idx, payload.src_station] += 1
            elif event_type == CitiBikeEvents.ReturnBike:
                payload: BikeReturnPayload = finished_events[self._next_event_idx].body
                supply_history[interval_idx, payload.to_station_idx] += payload.number

            # Update the index to the finished event that has not been processed.
            self._next_event_idx += 1

        self._num_recorded_interval += num_interval_to_record

        # Record to the forecasters.
        for i in range(self._num_station):
            self._demand_forecaster[i].record(demand_history[:, i])
            self._supply_forecaster[i].record(supply_history[:, i])

    def _forecast_demand_and_supply(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                The first item indicates the forecasting demand for each station in each time interval,
                with shape: (num_time_interval, num_station).
                The second item indicates the forecasting supply for each station in each time interval,
                with shape: (num_time_interval, num_station).
        """
        demand = (
            np.array(
                [round(self._demand_forecaster[i].forecast()) for i in range(self._num_station)],
                dtype=np.int16,
            )
            .reshape((1, -1))
            .repeat(self._num_time_interval, axis=0)
        )

        supply = (
            np.array(
                [round(self._supply_forecaster[i].forecast()) for i in range(self._num_station)],
                dtype=np.int16,
            )
            .reshape((1, -1))
            .repeat(self._num_time_interval, axis=0)
        )

        return demand, supply

    def __peep_at_the_future(self, env_tick: int):
        demand = np.zeros((self._num_time_interval, self._num_station), dtype=np.int16)
        supply = np.zeros((self._num_time_interval, self._num_station), dtype=np.int16)

        for tick in range(env_tick, env_tick + self._num_time_interval * self._ticks_per_interval):
            interval_idx = (tick - env_tick) // self._ticks_per_interval

            # Process to get the future demand and supply from TRIP_PICKER.
            for trip in TRIP_PICKER.items(tick=tick):
                demand[interval_idx, trip.src_station] += 1
                supply_interval_idx = (tick + trip.durations - env_tick) // self._ticks_per_interval
                if supply_interval_idx < self._num_time_interval:
                    supply[supply_interval_idx, trip.dest_station] += 1

            # Process to get the future supply from Pending Events.
            for pending_event in ENV.get_pending_events(tick=tick):
                if pending_event.event_type == CitiBikeEvents.ReturnBike:
                    payload: BikeReturnPayload = pending_event.body
                    supply[interval_idx, payload.to_station_idx] += payload.number

        return demand, supply

    # ============================= private end =============================

    def get_action_list(self, env_tick: int, init_inventory: np.ndarray, finished_events: List[AbsEvent]):
        if PEEP_AND_USE_REAL_DATA:
            demand, supply = self.__peep_at_the_future(env_tick=env_tick)
        else:
            self._record_history(env_tick=env_tick, finished_events=finished_events)
            demand, supply = self._forecast_demand_and_supply()

        transfer_list = self._ilp.get_transfer_list(
            env_tick=env_tick,
            init_inventory=init_inventory,
            demand=demand,
            supply=supply,
        )

        action_list = [
            Action(from_station_idx=item[0], to_station_idx=item[1], number=min(item[2], init_inventory[item[0]]))
            for item in transfer_list
        ]
        return action_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--peep",
        action="store_true",
        help="If set, peep the future demand and supply of bikes for each station directly from the log data.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="examples/citi_bike/online_lp/config.yml",
        help="The path of the config file.",
    )
    parser.add_argument(
        "-t",
        "--topology",
        type=str,
        help="Which topology to use. If set, it will over-write the topology set in the config file.",
    )
    parser.add_argument(
        "-r",
        "--seed",
        type=int,
        help="The random seed for the environment. If set, it will over-write the seed set in the config file.",
    )
    args = parser.parse_args()

    # Read the configuration.
    with io.open(args.config, "r") as in_file:
        raw_config = yaml.safe_load(in_file)
        config = convert_dottable(raw_config)

    # Overwrite the config.
    if args.topology is not None:
        config.env.topology = args.topology
    if args.seed is not None:
        config.env.seed = args.seed
    if args.peep:
        PEEP_AND_USE_REAL_DATA = True

    # Init an environment for Citi Bike.
    env = Env(
        scenario=config.env.scenario,
        topology=config.env.topology,
        start_tick=config.env.start_tick,
        durations=config.env.durations,
        snapshot_resolution=config.env.resolution,
    )

    # For debug only, used to peep the BE to get the real future data.
    if PEEP_AND_USE_REAL_DATA:
        ENV = env
        TRIP_PICKER = BinaryReader(env.configs["trip_data"]).items_tick_picker(
            start_time_offset=config.env.start_tick,
            end_time_offset=(config.env.start_tick + config.env.durations),
            time_unit="m",
        )

    if config.env.seed is not None:
        env.set_seed(config.env.seed)

    # Start simulation.
    decision_event: DecisionEvent = None
    action: Action = None
    is_done: bool = False
    _, decision_event, is_done = env.step(action=None)

    # TODO: Update the Env interface.
    num_station = len(env.agent_idx_list)
    station_distance_adj = np.array(
        load_adj_from_csv(env.configs["distance_adj_data"], skiprows=1),
    ).reshape(num_station, num_station)
    station_neighbor_list = [neighbor_list[1:] for neighbor_list in np.argsort(station_distance_adj, axis=1).tolist()]

    # Init a Moving-Average based ILP agent.
    decision_interval = env.configs["decision"]["resolution"]
    ilp = CitiBikeILP(
        num_station=num_station,
        num_neighbor=min(config.ilp.num_neighbor, num_station - 1),
        station_capacity=env.snapshot_list["stations"][env.frame_index : env.agent_idx_list : "capacity"],
        station_neighbor_list=station_neighbor_list,
        decision_interval=decision_interval,
        config=config.ilp,
    )
    agent = MaIlpAgent(
        ilp=ilp,
        num_station=num_station,
        num_time_interval=math.ceil(config.ilp.plan_window_size / decision_interval),
        ticks_per_interval=decision_interval,
        ma_window_size=config.forecasting.ma_window_size,
    )

    pre_decision_tick: int = -1
    while not is_done:
        if decision_event.tick == pre_decision_tick:
            action = None
        else:
            action = agent.get_action_list(
                env_tick=env.tick,
                init_inventory=env.snapshot_list["stations"][env.frame_index : env.agent_idx_list : "bikes"].astype(
                    np.int16,
                ),
                finished_events=env.get_finished_events(),
            )
            pre_decision_tick = decision_event.tick
        _, decision_event, is_done = env.step(action=action)

    print(
        f"[{'De' if PEEP_AND_USE_REAL_DATA else 'MA'}] "
        f"Topology {config.env.topology} with seed {config.env.seed}: {env.metrics}",
    )
