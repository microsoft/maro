import math
from typing import List, Tuple

import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, value, GLPK

from maro.utils import DottableDict

class LP():
    def __init__(
        self, num_station: int, num_neighbor: int,
        station_capacity: List[int], station_neighbor_list: List[List[int]],
        decision_interval: int, config: DottableDict
    ):
        """A simple Linear Programming formulation for solving the bike repositioning problem.

        Args:
            num_station (int): Number of stations in current topology.
            num_neighbor (int): Number of neighbors that needed to consider when repositioning.
            station_capacity (List[int]): The capacity of each station. Length: num_station.
            station_neighbor_list (List[List[int]]): The neighbor station of each station. Size: (num_station, num_station).
            decision_interval (int): The time interval of the environment to check and trigger a decision_event.
            config (DottableDict): LP configuration. Including the configuration for
                problem formulation, objectives, solution application.
        """
        self._num_station = num_station
        self._num_neighbor = num_neighbor
        self._station_capacity = station_capacity
        self._station_neighbor_list = station_neighbor_list
        self._decision_interval = decision_interval

        self._plan_window_size = config.plan_window_size
        self._apply_buffer_size = config.apply_buffer_size
        self._safety_inventory_limit = config.safety_inventory_limit

        self._fulfillment_time_decay_factor = config.objective.fulfillment_time_decay_factor
        self._transfer_cost_factor = config.objective.transfer_cost_factor
        self._safety_inventory_reward_factor = config.objective.safety_inventory_reward_factor

        self._num_tick = math.ceil(self._plan_window_size / self._decision_interval)
        self._last_start_tick = -1

    def _init_variables(self, init_inventory: np.ndarray):
        self._inventory = np.zeros((self._num_tick, self._num_station)).tolist()
        self._safety_inventory = np.zeros((self._num_tick, self._num_station)).tolist()
        self._fulfillment = np.zeros((self._num_tick, self._num_station)).tolist()
        self._transfer = np.zeros((self._num_tick, self._num_station, self._num_neighbor)).tolist()

        # The intermediate variables for clearer constraints.
        self._transfer_from = np.zeros((self._num_tick, self._num_station)).tolist()
        self._transfer_to = np.zeros((self._num_tick, self._num_station)).tolist()

        for tick in range(self._num_tick):
            for station in range(self._num_station):
                self._inventory[tick][station] = LpVariable(
                    name=f"T{tick}_S{station}_Inv", lowBound=0, upBound=self._station_capacity[station], cat="Integer"
                )
                self._safety_inventory[tick][station] = LpVariable(
                    name=f"T{tick}_S{station}_SafetyInv",
                    lowBound=0, upBound=(self._safety_inventory_limit * self._station_capacity[station]), cat="Integer"
                )
                self._fulfillment[tick][station] = LpVariable(
                    name=f"T{tick}_S{station}_Fulfillment", lowBound=0, cat="Integer"
                )

                # For intermediate variables.
                self._transfer_from[tick][station] = LpVariable(
                    name=f"T{tick}_TransferFrom{station}", lowBound=0, cat="Integer"
                )
                self._transfer_to[tick][station] = LpVariable(
                    name=f"T{tick}_TransferTo{station}", lowBound=0, cat="Integer"
                )

                for neighbor_idx in range(self._num_neighbor):
                    self._transfer[tick][station][neighbor_idx] = LpVariable(
                        name=f"T{tick}_Transfer_from{station}_to{neighbor_idx}th", lowBound=0, cat="Integer"
                    )

        # Initialize the current inventory.
        for station in range(self._num_station):
            self._inventory[0][station] = init_inventory[station]

    def _add_constraints(self, problem: LpProblem, demand: np.ndarray, supply: np.ndarray):
        for tick in range(self._num_tick):
            for station in range(self._num_station):
                problem += (
                    self._fulfillment[tick][station] <= demand[tick, station]
                ), f"Fulfillment_Limit_T{tick}_S{station}"
                # For intermediate variables.
                problem += (
                    self._transfer_from[tick][station] == lpSum(
                        self._transfer[tick][station][neighbor_idx] for neighbor_idx in range(self._num_neighbor)
                    )
                ), f"TotalTransferFrom_T{tick}_S{station}"
                problem += (
                    self._transfer_to[tick][station] == lpSum(
                        self._transfer[tick][neighbor][self._station_neighbor_list[neighbor].index(station)]
                        for neighbor in range(self._num_station)
                        if station in self._station_neighbor_list[neighbor][:self._num_neighbor]
                    )
                ), f"TotalTransferTo_T{tick}_S{station}"

        for tick in range(1, self._num_tick):
            for station in range(self._num_station):
                problem += (
                    self._inventory[tick][station] == (
                        self._inventory[tick - 1][station]
                        + supply[tick - 1][station] - self._fulfillment[tick - 1][station]
                        + self._transfer_to[tick - 1][station] - self._transfer_from[tick - 1][station]
                    )
                ), f"Inventory_T{tick}_S{station}"
                problem += (
                    self._safety_inventory[tick][station] <= (
                        self._inventory[tick - 1][station]
                        + supply[tick - 1][station] - self._fulfillment[tick - 1][station]
                        - self._transfer_from[tick - 1][station]
                    )
                ), f"SafetyInventory_T{tick}_S{station}"

    def _set_objective(self, problem: LpProblem):
        fulfillment_gain = lpSum([
            math.pow(self._fulfillment_time_decay_factor, tick) * lpSum([
                self._fulfillment[tick][station] for station in range(self._num_station)
            ]) for tick in range(self._num_tick)
        ])

        safety_inventory_reward = self._safety_inventory_reward_factor * lpSum([
            self._safety_inventory[tick][station] for station in range(self._num_station)
            for tick in range(self._num_tick)
        ])

        transfer_cost = self._transfer_cost_factor * lpSum([
            self._transfer_to[tick][station] for station in range(self._num_station)
            for tick in range(self._num_tick)
        ])

        problem += (fulfillment_gain + safety_inventory_reward - transfer_cost)

    def _formulate_and_solve(
        self, env_tick: int, init_inventory: np.ndarray, demand: np.ndarray, supply: np.ndarray
    ):
        problem = LpProblem(
            name = f"Citi_Bike_Repositioning_from_tick_{env_tick}",
            sense=LpMaximize,
        )
        self._init_variables(init_inventory=init_inventory)
        self._add_constraints(problem=problem, demand=demand, supply=supply)
        self._set_objective(problem=problem)
        problem.solve()

    def get_transfer_list(
        self, env_tick: int, init_inventory: np.ndarray, demand: np.ndarray, supply: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Get the transfer list for the given env_tick.

        Args:
            env_tick (int): The environment tick when calling this function.
            init_inventory (np.ndarray): The initial inventory of each station.
                Shape: (num_station).
            demand (np.ndarray): The demand for each station in each time interval.
                Shape: (num_time_interval, num_station).
            supply (np.ndarray): The supply for each station in each time interval.
                Shape: (num_time_interval, num_station).

        Returns:
            List[Tuple[int, int, int]]: The solution for current env_tick.
            The element in each Tuple represents the station, neighbor, and number to transfer respectively.
        """
        if env_tick >= self._last_start_tick + self._apply_buffer_size:
            self._last_start_tick = env_tick
            self._formulate_and_solve(
                env_tick=env_tick, init_inventory=init_inventory, demand=demand, supply=supply
            )

        tick = (env_tick - self._last_start_tick) // self._decision_interval
        transfer_list = []
        for station in range(self._num_station):
            for neighbor_idx in range(self._num_neighbor):
                num = self._transfer[tick][station][neighbor_idx].varValue
                neighbor = self._station_neighbor_list[station][neighbor_idx]
                if num > 0:
                    transfer_list.append((station, neighbor, num))

        return transfer_list
