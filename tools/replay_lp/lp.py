# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import math, os, re
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, value
from maro.utils import Logger, LogFormat

def get_num(varname):
    if isinstance(varname, int):
        return varname
    else:
        return varname.varValue

class LPReplayer():
    def __init__(self,
                 configs,
                 log_folder: str,
                 port_list: list,
                 vessel_list: list,
                 port_capacity: dict,
                 vessel_capacity: dict,
                 vessel_routes: dict,
                 full_return_buffer_ticks: dict,
                 empty_return_buffer_ticks: dict,
                 orders: dict = None,
                 tick_vessel_port_connection: dict = None,
                 ):
        self._configs = configs
        self._log_folder = log_folder
        self._file_prefix = configs.dump_file_prefix

        # How many steps we think in LP
        self._window_size = configs.plan_window_size

        self._time_decay_factor = configs.time_decay.time_decay_factor
        self._enable_time_decay = configs.time_decay.enable

        # Constant value, assigned with initial parameters directly
        self._ports = port_list
        self._vessels = vessel_list
        self._port_capacity = port_capacity
        self._vessel_capacity = vessel_capacity
        self._route_dict = vessel_routes

        # Expected value of random variables
        self._full_return_buffer_ticks = full_return_buffer_ticks
        self._empty_return_buffer_ticks = empty_return_buffer_ticks

        # Info can be overwritten when calling choose_action()
        self._tick_orders = orders
        self._tick_vessel_port_connection = tick_vessel_port_connection

        self._orders= dict()
        self._step_vessel_port_connection = list()

    def _set_step_orders(self, current_tick, tick_orders=None):
        if tick_orders:
            self._tick_orders = tick_orders
        self._orders.clear()
        for tick, orders in self._tick_orders.items():
            if tick >= current_tick and tick < current_tick + self._window_size:
                self._orders[tick - current_tick] = orders

    def _set_step_vessel_port_connection(self, current_tick, tick_vessel_port_connection=None):
        if tick_vessel_port_connection:
            self._tick_vessel_port_connection = tick_vessel_port_connection
        self._step_vessel_port_connection.clear()
        for tick in range(current_tick, current_tick + self._window_size):
            step = tick - current_tick
            self._step_vessel_port_connection.append({vessel: None for vessel in self._vessels})
            if tick in self._tick_vessel_port_connection:
                for vessel, port in self._tick_vessel_port_connection[tick].items():
                    self._step_vessel_port_connection[step][vessel] = port

    def _find_next_coming_vessel(self, order_step, src_port, dest_port):
        reachable_vessel_list = [vessel for vessel in self._vessels if src_port in self._route_dict[vessel] and dest_port in self._route_dict[vessel]]
        
        for step in range(order_step + 1, self._window_size):
            for vessel in reachable_vessel_list:
                if self._step_vessel_port_connection[step][vessel] == src_port:
                    return step, vessel
        
        return None, None

    # Decision Variables
    def _init_variables(self,
                        initial_port_empty: dict,
                        initial_port_on_consignee: dict,
                        initial_port_full: dict,
                        initial_vessel_empty: dict,
                        initial_vessel_full: dict
                        ):

        # Initialize decision variables list
        # corresponding to order
        self._order_apply = [{} for i in range(self._window_size)]      # [s][p1][p2]
        self._port_full = [{} for i in range(self._window_size)]   # [s][p1][p2]
        self._full_has_loaded = [{} for i in range(self._window_size)]
        
        # corresponding to valid (port, vessel)
        self._load_full = [{} for i in range(self._window_size)]   # [s][p1][p2][v]
        self._discharge_full = [{} for i in range(self._window_size)] # [s][p2][v]
        
        self._load_empty = [{} for i in range(self._window_size)]   # [s][p][v]
        self._discharge_empty = [{} for i in range(self._window_size)] # [s][p][v]
        
        # corresponding to time
        self._port_empty = [{} for i in range(self._window_size)]       # [s][p], empty inventory
        self._port_on_consignee = [{} for i in range(self._window_size)]    # [s][p], on_consignee increment
        
        self._vessel_empty = [{} for i in range(self._window_size)] # [s][v], empty inventory
        self._vessel_full = [{} for i in range(self._window_size)] # [s][v][p2], full inventory

        # Initialize decision variables related to port & vessel keys
        for s in range(self._window_size):
            for p in self._ports:
                self._port_empty[s][p] = 0
                self._port_on_consignee[s][p] = 0

                self._order_apply[s][p] = dict()
                self._port_full[s][p] = dict()
                self._full_has_loaded[s][p] = dict()
                
                self._load_full[s][p] = dict()
                self._discharge_full[s][p] = dict()
                
                self._load_empty[s][p] = dict()
                self._discharge_empty[s][p] = dict()

                for v in self._vessels:
                    self._discharge_full[s][p][v] = 0
                    self._load_empty[s][p][v] = 0
                    self._discharge_empty[s][p][v] = 0
                
                for p2 in self._ports:
                    self._order_apply[s][p][p2] = 0
                    self._port_full[s][p][p2] = 0
                    self._full_has_loaded[s][p][p2] = True

                    self._load_full[s][p][p2] = dict()
                    for v in self._vessels:
                        self._load_full[s][p][p2][v] = 0

            for v in self._vessels:
                self._vessel_empty[s][v] = 0

                self._vessel_full[s][v] = dict()
                for p in self._ports:
                    self._vessel_full[s][v][p] = 0

        # Initial Inventory
        for p in self._ports:
            self._port_empty[0][p] = initial_port_empty[p]
            # TODO: not support if empty return buffer tick > 1
            self._port_on_consignee[0][p] = initial_port_on_consignee[p]
            for p2 in self._ports:
                self._port_full[0][p][p2] = initial_port_full[p][p2]
        for v in self._vessels:
            self._vessel_empty[0][v] = initial_vessel_empty[v]
            for p2 in self._ports:
                self._vessel_full[0][v][p2] = initial_vessel_full[v][p2]

        for s in range(1, self._window_size):
            for p in self._ports:
                # Port Empty Inventory
                self._port_empty[s][p] = LpVariable(name=f'Port_Empty_Inventory_{s}_{p}', lowBound=0, cat='Integer')
                vessel_list = []
                for vessel_code, port_code in self._step_vessel_port_connection[s].items():
                    if port_code == p:
                        vessel_list.append(vessel_code)
                # On-Consignee Increment
                if vessel_list:
                    self._port_on_consignee[s][p] = LpVariable(name=f'Port_On_Consignee_Increment_{s}_{p}', lowBound=0, cat='Integer')
            for v in self._vessels:
                # Vessel Empty Inventory
                self._vessel_empty[s][v] = LpVariable(name=f'Vessel_Empty_Inventory_{s}_{v}', lowBound=0, cat='Integer')
                # Vessel Full Inventory
                for p2 in self._ports:
                    self._vessel_full[s][v][p2] = LpVariable(name=f'Vessel_Full_Inventory_{s}_{v}_{p2}', lowBound=0, cat='Integer')

        for s in range(self._window_size):
            # For each decision point
            for v, p in self._step_vessel_port_connection[s].items():
                if p != None:
                    # Number of Empty loaded from port to vessel
                    self._load_empty[s][p][v] = LpVariable(name=f'Load_Empty_{s}_{p}_{v}', lowBound=0, cat='Integer')
                    # Number of Empty dischargeed from vessel to port
                    self._discharge_empty[s][p][v] = LpVariable(name=f'Discharge_Empty_{s}_{p}_{v}', lowBound=0, cat='Integer')
                    # Number of Full dischargeed from vessel to port
                    self._discharge_full[s][p][v] = LpVariable(name=f'Discharge_Full_{s}_{p}_{v}', lowBound=0, cat='Integer')

        # For each order
        for s in self._orders.keys():
            for p1 in self._orders[s].keys():
                for p2 in self._orders[s][p1].keys():
                    if self._orders[s][p1][p2] > 0:
                        # How much Empty we can pay for this order
                        self._order_apply[s][p1][p2] = LpVariable(name=f'Order_Apply_Quantity_{s}_{p1}_{p2}', lowBound=0, cat='Integer')
                        s_lo = s + self._full_return_buffer_ticks[p1]
                        if s_lo < self._window_size:
                            self._port_full[s_lo][p1][p2] = LpVariable(name=f'Port_Full_{s_lo}_{p1}_{p2}', lowBound=0, cat='Integer')
                            # Find the next coming vessel that can transport this order to its destination
                            self._full_has_loaded[s_lo][p1][p2] = False
                            vessel_arrive_time, vessel_code = self._find_next_coming_vessel(s_lo, p1, p2)
                            # Number of full loaded from port p1 to vessel, with port p2 as the destination
                            if vessel_arrive_time and vessel_arrive_time < self._window_size: # can load full to vessel
                                self._load_full[vessel_arrive_time][p1][p2][vessel_code] = \
                                    LpVariable(name=f'Load_Full_{vessel_arrive_time}_{p1}_{p2}_{vessel_code}', lowBound=0, cat='Integer')

    def _set_objective(self, problem):
        order_gain_factor = [math.pow(self._time_decay_factor, s) for s in range(self._window_size)] if self._enable_time_decay else [1] * self._window_size

        problem += lpSum([order_gain_factor[s] * self._order_apply[s][p1][p2] \
            for s in range(self._window_size) \
                for p1 in self._order_apply[s].keys() \
                    for p2 in self._order_apply[s][p1].keys() \
                        ])

    def _add_constraints(self, problem):
        # Order processing
        for s in self._orders.keys():
            for p1 in self._orders[s].keys():
                for p2 in self._orders[s][p1].keys():
                    # Order apply: operation constraint 1/2
                    problem += self._order_apply[s][p1][p2] <= self._orders[s][p1][p2]

                    # Port Full:
                    s_lo = s + self._full_return_buffer_ticks[p1]
                    if s_lo < self._window_size:
                        problem += self._order_apply[s][p1][p2] == self._port_full[s_lo][p1][p2]

        # Load Full: operation constraint
        for s in self._orders.keys():
            for p1 in self._orders[s].keys():
                for p2 in self._orders[s][p1].keys():
                    s_lo = s + self._full_return_buffer_ticks[p1]
                    vessel_arrive_time, vessel_code = self._find_next_coming_vessel(s_lo, p1, p2)
                    start_time = s_lo if s == 0 else s_lo + 1
                    if not vessel_arrive_time or not vessel_arrive_time < self._window_size or self._full_has_loaded[start_time][p1][p2]:
                        continue
                    problem += self._load_full[vessel_arrive_time][p1][p2][vessel_code] == \
                        lpSum([self._port_full[si][p1][p2] for si in range(start_time, vessel_arrive_time + 1)])
                    for time_before_arrive in range(start_time, vessel_arrive_time + 1):
                        self._full_has_loaded[time_before_arrive][p1][p2] = True

        # Discharge Full: operation constraint
        for s in range(self._window_size):
            for v, p in self._step_vessel_port_connection[s].items():
                if p != None:
                    problem += self._discharge_full[s][p][v] == self._vessel_full[s][v][p]

        # Port On-Consignee:
        for s in range(self._window_size):
            for p in self._ports:
                problem += self._port_on_consignee[s][p] == lpSum([self._discharge_full[s][p][v] for v in self._vessels])
        
        # Order apply: operation constraint 2/2
        # Load empty: operation constraint
        for s in range(self._window_size):
            for p in self._ports:
                s_li = s - self._empty_return_buffer_ticks[p]
                if s_li >= 0:
                    for vessel_idx in range(len(self._vessels)):
                        problem += self._port_empty[s][p] + self._port_on_consignee[s_li][p] \
                            - lpSum([self._order_apply[s][p][des_p] for des_p in self._ports]) \
                                + lpSum([self._discharge_empty[s][p][self._vessels[idx]] for idx in range(vessel_idx)]) \
                                    >= lpSum([self._load_empty[s][p][self._vessels[idx]] for idx in range(vessel_idx + 1)]) 
                else:
                    for vessel_idx in range(len(self._vessels)):
                        problem += self._port_empty[s][p] \
                            - lpSum([self._order_apply[s][p][des_p] for des_p in self._ports]) \
                                + lpSum([self._discharge_empty[s][p][self._vessels[idx]] for idx in range(vessel_idx)]) \
                                    >= lpSum([self._load_empty[s][p][self._vessels[idx]] for idx in range(vessel_idx + 1)]) 


        # Discharge Empty: operation constraint
        for s in range(self._window_size):
            for v, p in self._step_vessel_port_connection[s].items():
                if p != None:
                    problem += self._discharge_empty[s][p][v] <= self._vessel_empty[s][v]

        # Inventory of vessels
        for s in range(1, self._window_size):
            for v in self._vessels:
                pre_p = self._step_vessel_port_connection[s-1][v]
                # Empty: recursion formula
                if pre_p != None:
                    problem += self._vessel_empty[s][v] == self._vessel_empty[s-1][v] \
                        + self._load_empty[s-1][pre_p][v] - self._discharge_empty[s-1][pre_p][v]
                else:
                    problem += self._vessel_empty[s][v] == self._vessel_empty[s-1][v]
                # Full: recursion formula
                for p2 in self._ports:
                    if pre_p != None:
                        problem += self._vessel_full[s][v][p2] == self._vessel_full[s-1][v][p2] \
                            + self._load_full[s-1][pre_p][p2][v] - self._discharge_full[s-1][p2][v]
                    else:
                        problem += self._vessel_full[s][v][p2] == self._vessel_full[s-1][v][p2]
        
        # Inventory of ports
        for s in range(1, self._window_size):
            for p in self._ports:
                # Empty: recursion formula
                s_li = s - 1 - self._empty_return_buffer_ticks[p]
                if s_li >= 0:
                    problem += self._port_empty[s][p] == self._port_empty[s-1][p] \
                        + lpSum([self._discharge_empty[s-1][p][v] - self._load_empty[s-1][p][v] for v in self._vessels]) \
                            - lpSum([self._order_apply[s-1][p][des_p] for des_p in self._ports]) \
                                + self._port_on_consignee[s_li][p]
                else:
                    problem += self._port_empty[s][p] == self._port_empty[s-1][p] \
                        + lpSum([self._discharge_empty[s-1][p][v] - self._load_empty[s-1][p][v] for v in self._vessels]) \
                            - lpSum([self._order_apply[s-1][p][des_p] for des_p in self._ports])

        # Capacity constraint
        for s in range(self._window_size):
            # Vessel
            for v in self._vessels:
                problem += self._vessel_capacity[v] >= self._vessel_empty[s][v] \
                    + lpSum([self._vessel_full[s][v][des_p] for des_p in self._ports])

    def _shortage_calculation(self):
        self._shortage = [0] * self._window_size
        self._total_shortage = 0
        for step in self._orders.keys():
            self._shortage[step] = 0
            for p1 in self._orders[step].keys():
                for port2 in self._orders[step][p1].keys():
                    self._shortage[step] += self._orders[step][p1][port2] - self._order_apply[step][p1][port2].varValue
            self._total_shortage += self._shortage[step]

    def _write_solution_to_file(self):        
        solution_logger = Logger(tag=f'{self._file_prefix}', format_=LogFormat.none,
            dump_folder=self._log_folder, dump_mode='w', auto_timestamp=False, extension_name='txt')

        for step in range(self._window_size):
            solution_logger.debug(f'******************** Tick {step} ********************')
            for port in self._ports:
                solution_logger.debug(f'[{port}]')
                solution_logger.debug(f'    Empty Inventory: {get_num(self._port_empty[step][port])}')
                solution_logger.debug(f'    On-Consignee Increment: {get_num(self._port_on_consignee[step][port])}')
                if step in self._orders.keys() and port in self._orders[step].keys():
                    for p2 in self._orders[step][port].keys():
                        solution_logger.debug(f'    Order Apply to {p2}: {get_num(self._order_apply[step][port][p2])} / {self._orders[step][port][p2]}')
                for vessel in self._vessels:
                    if self._step_vessel_port_connection[step][vessel] == port:
                        solution_logger.debug(f'    [{vessel}]')
                        solution_logger.debug(f'        Empty Inventory: {get_num(self._vessel_empty[step][vessel])}')
                        for p2 in self._ports:
                            if get_num(self._load_full[step][port][p2][vessel]) > 0:
                                solution_logger.debug(f'        Load Full for {p2}: {get_num(self._load_full[step][port][p2][vessel])}')
                        solution_logger.debug(f'        Discharge Full: {get_num(self._discharge_full[step][port][vessel])}')
                        solution_logger.debug(f'        Load Empty: {get_num(self._load_empty[step][port][vessel])}')
                        solution_logger.debug(f'        Discharge Empty: {get_num(self._discharge_empty[step][port][vessel])}')
            solution_logger.debug(f'Sailing vessels')
            for vessel in self._vessels:
                if self._step_vessel_port_connection[step][vessel] == None:
                    solution_logger.debug(f'[{vessel}]')
                    solution_logger.debug(f'    Empty Inventory: {get_num(self._vessel_empty[step][vessel])}')
                    for p2 in self._ports:
                        if get_num(self._vessel_full[step][vessel][p2]) > 0:
                            solution_logger.debug(f'    Full for {p2}: {get_num(self._vessel_full[step][vessel][p2])}')
        solution_logger.info(f'******************************************************************************************')
        solution_logger.info(f'Status: {self._solution_status}')
        solution_logger.info(f'Objective: {self._objective_gotten}')
        solution_logger.info(f'Total shortage: {self._total_shortage}')

    def formulate_and_solve(self,
                            current_tick: int,
                            initial_port_empty: dict,
                            initial_port_on_consignee: dict,
                            initial_port_full: dict,
                            initial_vessel_empty: dict,
                            initial_vessel_full: dict,
                            tick_orders: dict = None,
                            tick_vessel_port_connection: dict = None
                            ):
        self._set_step_orders(current_tick=current_tick, tick_orders=tick_orders)
        self._set_step_vessel_port_connection(current_tick=current_tick, tick_vessel_port_connection=tick_vessel_port_connection)

        problem = LpProblem(name=f"ecr Problem: from Tick_{current_tick}", sense=LpMaximize)
        self._init_variables(initial_port_empty=initial_port_empty,
                             initial_port_on_consignee=initial_port_on_consignee,
                             initial_port_full=initial_port_full,
                             initial_vessel_empty=initial_vessel_empty,
                             initial_vessel_full=initial_vessel_full
                             )
        self._set_objective(problem)
        self._add_constraints(problem)
        problem.solve()

        lp_file_path = os.path.join(self._log_folder, f'{self._file_prefix}.lp')
        problem.writeLP(lp_file_path)

        assert (problem.status == 1)
        self._solution_status = LpStatus[problem.status]
        self._objective_gotten = value(problem.objective)
        self._shortage_calculation()

        # Show the details of the lp solution
        self._write_solution_to_file()

    def choose_action(self,
                      current_tick: int,
                      port_code: str,
                      vessel_code: str,
                      initial_port_empty: dict = None,
                      initial_port_on_consignee: dict = None,
                      initial_port_full: dict = None,
                      initial_vessel_empty: dict = None,
                      initial_vessel_full: dict = None,
                      tick_orders: dict = None,
                      tick_vessel_port_connection: dict = None
                      ):
        assert current_tick < self._window_size

        step = current_tick

        num_load_empty = get_num(self._load_empty[step][port_code][vessel_code])
        num_discharge_empty = get_num(self._discharge_empty[step][port_code][vessel_code])

        # Execute Action
        return num_discharge_empty - num_load_empty
    
    def clear(self):
        self._tick_orders.clear()
        self._tick_vessel_port_connection.clear()