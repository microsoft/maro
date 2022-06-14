# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import List, Tuple

import numpy as np
from pulp import PULP_CBC_CMD, LpInteger, LpMaximize, LpProblem, LpVariable, lpSum

from maro.utils import DottableDict


class CitiBikeILP:
    def __init__(
        self,
        num_station: int,
        num_neighbor: int,
        station_capacity: List[int],
        station_neighbor_list: List[List[int]],
        decision_interval: int,
        config: DottableDict,
    ):
        """A simple Linear Programming formulation for solving the bike repositioning problem.

        Args:
            num_station (int): Number of stations in current topology.
            num_neighbor (int): Number of neighbors that needed to consider when repositioning.
            station_capacity (List[int]): The capacity of each station. Length: num_station.
            station_neighbor_list (List[List[int]]): The neighbor station of each station.
                Size: (num_station, num_station).
            decision_interval (int): The time interval of the environment to check and trigger a decision_event.
            config (DottableDict): ILP configuration. Including the configuration for
                problem formulation, objectives, solution application.
        """
        self._num_station = num_station
        self._num_neighbor = num_neighbor
        self._station_capacity = station_capacity
        self._station_neighbor_list = station_neighbor_list
        self._decision_interval = decision_interval

        # Plan Window Size (Unit: Env tick): The time length of the ILP problem formulation.
        self._plan_window_size = config.plan_window_size
        # Apply Buffer Size (Unit: Env tick): The time length of the ILP problem solution that will be applied to the
        # environment. Valid Apply Buffer Size should not be longer than Plan Window Size.
        self._apply_buffer_size = min(config.apply_buffer_size, self._plan_window_size)
        # The upper bound of the Safety Inventory that can gain reward in objective.
        self._safety_inventory_limit = config.safety_inventory_limit

        self._fulfillment_time_decay_factor = config.objective.fulfillment_time_decay_factor
        self._transfer_cost_factor = config.objective.transfer_cost_factor
        self._safety_inventory_reward_factor = config.objective.safety_inventory_reward_factor
        self._safety_inventory_reward_time_decay_factor = config.objective.safety_inventory_reward_time_decay_factor

        self._num_decision_point = math.ceil(self._plan_window_size / self._decision_interval)
        self._last_start_tick = -1

    # ============================= private start =============================

    def _init_variables(self, init_inventory: np.ndarray):
        def _init_with_shape(shape: tuple):
            return np.zeros(shape, dtype=np.int16).tolist()

        self._inventory = _init_with_shape(shape=(self._num_decision_point, self._num_station))
        self._safety_inventory = _init_with_shape(shape=(self._num_decision_point, self._num_station))
        self._fulfillment = _init_with_shape(shape=(self._num_decision_point, self._num_station))
        self._transfer = _init_with_shape(shape=(self._num_decision_point, self._num_station, self._num_neighbor))

        # The intermediate variables for clearer constraints.
        self._transfer_from = _init_with_shape(shape=(self._num_decision_point, self._num_station))
        self._transfer_to = _init_with_shape(shape=(self._num_decision_point, self._num_station))

        for decision_point in range(self._num_decision_point):
            for station in range(self._num_station):
                self._inventory[decision_point][station] = LpVariable(
                    name=f"T{decision_point}_S{station}_Inv",
                    lowBound=0,
                    upBound=self._station_capacity[station],
                    cat=LpInteger,
                )
                self._safety_inventory[decision_point][station] = LpVariable(
                    name=f"T{decision_point}_S{station}_SafetyInv",
                    lowBound=0,
                    upBound=round(self._safety_inventory_limit * self._station_capacity[station]),
                    cat=LpInteger,
                )
                self._fulfillment[decision_point][station] = LpVariable(
                    name=f"T{decision_point}_S{station}_Fulfillment",
                    lowBound=0,
                    cat=LpInteger,
                )

                # For intermediate variables.
                self._transfer_from[decision_point][station] = LpVariable(
                    name=f"T{decision_point}_TransferFrom{station}",
                    lowBound=0,
                    cat=LpInteger,
                )
                self._transfer_to[decision_point][station] = LpVariable(
                    name=f"T{decision_point}_TransferTo{station}",
                    lowBound=0,
                    cat=LpInteger,
                )

                for neighbor_idx in range(self._num_neighbor):
                    self._transfer[decision_point][station][neighbor_idx] = LpVariable(
                        name=f"T{decision_point}_Transfer_from{station}_to{neighbor_idx}th",
                        lowBound=0,
                        cat=LpInteger,
                    )

        # Initialize inventory of the first decision point with the environment's current inventory.
        for station in range(self._num_station):
            self._inventory[0][station] = init_inventory[station]

    def _add_constraints(self, problem: LpProblem, demand: np.ndarray, supply: np.ndarray):
        for decision_point in range(self._num_decision_point):
            for station in range(self._num_station):
                problem += (
                    self._fulfillment[decision_point][station] <= demand[decision_point, station]
                ), f"Fulfillment_Limit_T{decision_point}_S{station}"
                # For intermediate variables.
                problem += (
                    self._transfer_from[decision_point][station]
                    == lpSum(
                        self._transfer[decision_point][station][neighbor_idx]
                        for neighbor_idx in range(self._num_neighbor)
                    )
                ), f"TotalTransferFrom_T{decision_point}_S{station}"
                problem += (
                    self._transfer_to[decision_point][station]
                    == lpSum(
                        self._transfer[decision_point][neighbor][self._station_neighbor_list[neighbor].index(station)]
                        for neighbor in range(self._num_station)
                        if station in self._station_neighbor_list[neighbor][: self._num_neighbor]
                    )
                ), f"TotalTransferTo_T{decision_point}_S{station}"

        for decision_point in range(1, self._num_decision_point):
            for station in range(self._num_station):
                problem += (
                    self._inventory[decision_point][station]
                    == (
                        self._inventory[decision_point - 1][station]
                        + supply[decision_point - 1, station]
                        - self._fulfillment[decision_point - 1][station]
                        + self._transfer_to[decision_point - 1][station]
                        - self._transfer_from[decision_point - 1][station]
                    )
                ), f"Inventory_T{decision_point}_S{station}"
                problem += (
                    self._safety_inventory[decision_point][station]
                    <= (
                        self._inventory[decision_point - 1][station]
                        + supply[decision_point - 1, station]
                        - self._fulfillment[decision_point - 1][station]
                        - self._transfer_from[decision_point - 1][station]
                    )
                ), f"SafetyInventory_T{decision_point}_S{station}"

    def _set_objective(self, problem: LpProblem):
        fulfillment_gain = lpSum(
            math.pow(self._fulfillment_time_decay_factor, decision_point)
            * lpSum(self._fulfillment[decision_point][station] for station in range(self._num_station))
            for decision_point in range(self._num_decision_point)
        )

        safety_inventory_reward = self._safety_inventory_reward_factor * lpSum(
            math.pow(self._safety_inventory_reward_time_decay_factor, decision_point)
            * lpSum(self._safety_inventory[decision_point][station] for station in range(self._num_station))
            for decision_point in range(self._num_decision_point)
        )

        transfer_cost = self._transfer_cost_factor * lpSum(
            self._transfer_to[decision_point][station]
            for station in range(self._num_station)
            for decision_point in range(self._num_decision_point)
        )

        problem += fulfillment_gain + safety_inventory_reward - transfer_cost

    def _formulate_and_solve(
        self,
        env_tick: int,
        init_inventory: np.ndarray,
        demand: np.ndarray,
        supply: np.ndarray,
    ):
        problem = LpProblem(
            name=f"Citi_Bike_Repositioning_from_tick_{env_tick}",
            sense=LpMaximize,
        )
        self._init_variables(init_inventory=init_inventory)
        self._add_constraints(problem=problem, demand=demand, supply=supply)
        self._set_objective(problem=problem)
        problem.solve(PULP_CBC_CMD(msg=0))

    # ============================= private end =============================

    def get_transfer_list(
        self,
        env_tick: int,
        init_inventory: np.ndarray,
        demand: np.ndarray,
        supply: np.ndarray,
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
            List[Tuple[int, int, int]]: The solution for current env_tick. The element in each Tuple represents
            the source station, target station, and number to transfer respectively.
        """
        if env_tick >= self._last_start_tick + self._apply_buffer_size:
            self._last_start_tick = env_tick
            self._formulate_and_solve(
                env_tick=env_tick,
                init_inventory=init_inventory,
                demand=demand,
                supply=supply,
            )

        decision_point = (env_tick - self._last_start_tick) // self._decision_interval
        transfer_list = []
        for station in range(self._num_station):
            for neighbor_idx in range(self._num_neighbor):
                num = self._transfer[decision_point][station][neighbor_idx].varValue
                neighbor = self._station_neighbor_list[station][neighbor_idx]
                if num > 0:
                    transfer_list.append((station, neighbor, num))

        return transfer_list
