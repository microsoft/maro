import math

import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, GLPK, LpStatus

from maro.utils import DottableDict


class OnlineLP:
    """Solve the ECR problem using Online Linear Programming"""

    def __init__(
        self,
        port_idx2name: dict,
        vessel_idx2name: dict,
        topology_config: dict,
        lp_config: DottableDict,
    ):
        # topology related variables
        self._port_idx2name = port_idx2name
        self._vessel_idx2name = vessel_idx2name
        self._topology_config = topology_config

        self._port_name_list: list = list(self._port_idx2name.values())
        self._vessel_name_list: list = list(self._vessel_idx2name.values())

        # LP related parameters
        self.lp_config = lp_config

        # buffer related variables
        self._apply_buffer_length = self.lp_config.apply_buffer_length
        self._vessel_applied_buffer_times = {
            vessel_name: 0 for vessel_name in self._vessel_name_list
        }

        # LP related constants
        self._port_capacity: dict = {
            name: info["capacity"] for name, info in self._topology_config["ports"].items()
        }
        self._vessel_capacity: dict = {
            name: info["capacity"] for name, info in self._topology_config["vessels"].items()
        }

        self._vessel_full: dict = {}

        # LP related variables
        self._port_empty: dict = {}
        self._vessel_empty: dict = {}

        # LP related decision variables
        self._discharge_empty: dict = {}
        self._load_empty: dict = {}
        self._order_apply: dict = {}

        # Forecasted data
        self._orders: dict = {}
        self._return_empty: dict = {}
        self._vessel_arrival: dict = {}

        self._decision_step_list = None

    def _get_on_port_vessels(self, step: int, port_list: list):
        on_port_vessel_list: list = []

        if step >= 0:
            for vessel_name, port_name in self._vessel_arrival[step].items():
                if port_name in port_list:
                    on_port_vessel_list.append(vessel_name)

        return on_port_vessel_list

    def _init_variables(self):
        for wsize in range(-1, self.lp_config.window_size):
            self._port_empty[wsize] = dict()
            self._vessel_empty[wsize] = dict()
            self._vessel_full[wsize] = dict()
            self._order_apply[wsize] = dict()
            self._discharge_empty[wsize] = dict()
            self._load_empty[wsize] = dict()

            for pname in self._port_name_list:
                self._port_empty[wsize][pname] = LpVariable(
                    name=f"port_empty__{wsize}_{pname}",
                    lowBound=0,
                    cat="Integer",
                )
                self._order_apply[wsize][pname] = LpVariable(
                    name=f"order_apply__{wsize}_{pname}",
                    lowBound=0,
                    cat="Integer",
                )

            on_port_vessels = self._get_on_port_vessels(
                wsize, self._port_name_list
            )
            for vname in self._vessel_name_list:
                self._vessel_empty[wsize][vname] = LpVariable(
                    name=f"vessel_empty__{wsize}_{vname}",
                    lowBound=0,
                    cat="Integer",
                )
                if vname in on_port_vessels:
                    self._load_empty[wsize][vname] = LpVariable(
                        name=f"load_empty__{wsize}_{vname}",
                        lowBound=0,
                        cat="Integer",
                    )
                    self._discharge_empty[wsize][vname] = LpVariable(
                        name=f"discharge_empty__{wsize}_{vname}",
                        lowBound=0,
                        cat="Integer",
                    )
                else:
                    self._load_empty[wsize][vname] = 0
                    self._discharge_empty[wsize][vname] = 0

    def _init_inventory(
        self,
        initial_port_empty: dict = None,
        initial_vessel_empty: dict = None,
        initial_vessel_full: dict = None,
    ):
        """Initialize invetory"""
        for pname in self._port_name_list:
            self._port_empty[-1][pname] = initial_port_empty[pname]
            self._order_apply[-1][pname] = 0

        for vname in self._vessel_name_list:
            self._vessel_empty[-1][vname] = initial_vessel_empty[vname]
            self._vessel_full[-1][vname] = np.sum(
                [
                    initial_vessel_full[vname][pfull]
                    for pfull in initial_vessel_full[vname].keys()
                ]
            )

    def _add_constraints(self, problem):
        """Add constraints to LP solver"""
        for wsize in range(-1, self.lp_config.window_size - 1):
            # for port empty
            for pname in self._port_name_list:
                on_port_vessel_list = self._get_on_port_vessels(wsize, [pname])
                problem += self._port_empty[wsize + 1][pname] == (
                    self._port_empty[wsize][pname]
                    + self._return_empty[wsize][pname]
                    - self._order_apply[wsize][pname]
                    + lpSum(
                        [
                            self._discharge_empty[wsize][vname]
                            - self._load_empty[wsize][vname]
                            for vname in on_port_vessel_list
                        ]
                    )
                )

            # for vessel empty
            for vname in self._vessel_name_list:
                problem += self._vessel_empty[wsize + 1][vname] == (
                    self._vessel_empty[wsize][vname]
                    + self._load_empty[wsize][vname]
                    - self._discharge_empty[wsize][vname]
                )

        for wsize in range(0, self.lp_config.window_size):
            # for capacity
            for pname in self._port_name_list:
                problem += (
                    self._port_empty[wsize][pname]
                    <= self._port_capacity[pname]
                )
            for vname in self._vessel_name_list:
                problem += (
                    self._vessel_empty[wsize][vname]
                    + self._vessel_full[wsize - 1][vname]
                    <= self._vessel_capacity[vname]
                )

            # for apply order
            for pname in self._port_name_list:
                problem += (
                    self._order_apply[wsize][pname]
                    <= self._orders[wsize][pname]
                )

    def _set_objective(self, problem):
        """Define the objective function"""
        order_gain = self.lp_config.order_gain_factor * lpSum(
            [
                math.pow(self.lp_config.time_decay, wsize)
                * self._order_apply[wsize][pname]
                for wsize in range(self.lp_config.window_size)
                for pname in self._order_apply[wsize].keys()
            ]
        )

        transit_cost = self.lp_config.transit_cost_factor * lpSum(
            [
                self._vessel_empty[wsize][vname]
                for wsize in self._vessel_arrival
                for vname in self._vessel_name_list
                if self._vessel_arrival[wsize] is None
                and self._vessel_arrival[wsize][vname] is None
            ]
        )

        load_discharge_cost = (
            self.lp_config.load_discharge_cost_factor
            * lpSum(
                [
                    self._load_empty[wsize][vname]
                    + self._discharge_empty[wsize][vname]
                    for wsize in self._vessel_arrival
                    for vname in self._vessel_arrival[wsize].keys()
                ]
            )
        )

        problem += order_gain - transit_cost - load_discharge_cost

    def _parse_and_get_vessel_full_prediction(self, vessel_full_delta):
        for step in range(self.lp_config.window_size):
            for vessel_name in self._vessel_name_list:
                # TODO: no need for this if-statement
                if vessel_name in self._vessel_arrival[step].keys():
                    step_vessel_full = (
                        self._vessel_full[step-1][vessel_name]
                        + vessel_full_delta[vessel_name][self._vessel_arrival[step][vessel_name]]
                    )
                else:
                    step_vessel_full = self._vessel_full[step-1][vessel_name]

                vessel_capacity = self._topology_config["vessels"][vessel_name]['capacity']
                vessel_full_upper_bound = (
                    vessel_capacity - self._vessel_empty[-1][vessel_name]
                    if step == 0
                    else vessel_capacity
                )

                step_vessel_full = max(step_vessel_full, 0)
                step_vessel_full = min(step_vessel_full, vessel_full_upper_bound)
                self._vessel_full[step][vessel_name] = step_vessel_full

        return

    def _formulate_and_solve(
        self,
        finished_events: list,
        snapshot_list,
        current_tick: int,
        initial_port_empty: dict,
        initial_vessel_empty: dict,
        initial_vessel_full: dict,
        vessel_arrival_prediction: dict,
        order_prediction: dict,
        return_empty_prediction: dict,
        vessel_full_delta_prediction: dict,
    ):
        """Solution Module for the LP solver"""
        self._vessel_arrival = vessel_arrival_prediction
        self._orders = order_prediction
        self._return_empty = return_empty_prediction

        self._init_variables()
        self._init_inventory(
            initial_port_empty, initial_vessel_empty, initial_vessel_full
        )
        self._parse_and_get_vessel_full_prediction(vessel_full_delta_prediction)

        problem = LpProblem(
            name=f"Problem:_from_Tick_{current_tick}", sense=LpMaximize
        )

        self._add_constraints(problem)
        self._set_objective(problem)

        problem.solve(GLPK(msg=0))

        # reset buffer
        self._vessel_applied_buffer_times = {
            vessel_name: 0 for vessel_name in self._vessel_name_list
        }
        self._decision_step_list: dict = {
            vessel_name: list() for vessel_name in self._vessel_name_list
        }

        for wsize in self._vessel_arrival.keys():
            for vessel_name in self._vessel_arrival[wsize].keys():
                self._decision_step_list[vessel_name].append(wsize)
        for vessel_name in self._decision_step_list.keys():
            self._decision_step_list[vessel_name].sort()

    def choose_action(
        self,
        current_tick: int,
        vessel_code: str,
        finished_events: list,
        snapshot_list,
        initial_port_empty: dict = None,
        initial_vessel_empty: dict = None,
        initial_vessel_full: dict = None,
        vessel_arrival_prediction: dict = None,
        order_prediction: dict = None,
        return_empty_prediction: dict = None,
        vessel_full_delta_prediction: dict = None,
    ):
        decision_step = self._find_next_decision_step(vessel_code)
        if not decision_step or decision_step >= self._apply_buffer_length:
            self._formulate_and_solve(
                finished_events,
                snapshot_list,
                current_tick,
                initial_port_empty,
                initial_vessel_empty,
                initial_vessel_full,
                vessel_arrival_prediction,
                order_prediction,
                return_empty_prediction,
                vessel_full_delta_prediction
            )
            decision_step = self._find_next_decision_step(vessel_code)

        return (
            self._discharge_empty[decision_step][vessel_code].varValue
            - self._load_empty[decision_step][vessel_code].varValue
        )

    def _find_next_decision_step(self, vessel_code):
        """Get next decision step"""
        if not self._decision_step_list:
            return None

        if self._vessel_applied_buffer_times[vessel_code] < len(self._decision_step_list[vessel_code]):
            next_decision_step = self._decision_step_list[vessel_code][self._vessel_applied_buffer_times[vessel_code]]
        else:
            next_decision_step = self._apply_buffer_length
        self._vessel_applied_buffer_times[vessel_code] += 1

        return next_decision_step

    def reset(self):
        pass
