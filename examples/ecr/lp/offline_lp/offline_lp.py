import math, os, re
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, value, GLPK
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
        self._file_prefix = configs.file_prefix
        self._logger = Logger(tag=f'lp_replayer', format_=LogFormat.none,
            dump_folder=self._log_folder, dump_mode='w', auto_timestamp=False, extension_name='txt')

        self._load_action = configs.load_actions.enable
        self._action_path = configs.load_actions.action_path
        self._enable_loaded_assert = configs.load_actions.enable_assert
        if self._load_action:
            self._action_check_logger = Logger(tag=f'lp_action_check', format_=LogFormat.none,
                dump_folder=self._log_folder, dump_mode='w', auto_timestamp=False, extension_name='txt')

        self._order_gain_factor = configs.factor.order_gain_factor
        self._transit_cost_factor = configs.factor.transit_cost_factor
        self._load_discharge_cost_factor = configs.factor.load_discharge_cost_factor
        self._full_delayed_factor = configs.factor.full_delayed_factor

        # How many steps we think in LP
        self._window_size = configs.params.window_size

        # How many steps we store in buffer
        self._apply_buffer_size = configs.params.apply_buffer_size
        self._apply_buffer_end = 0
        self._applied_idx = dict()

        self._time_decay = configs.time_decay.time_decay_ratio
        self._enable_time_decay = configs.time_decay.enable

        # self._logger.debug('*********************** Order ***********************')
        # total_needs = [{port: 0 for port in port_list} for s in range(configs.params.window_size)]
        # for tick, order in orders.items():
        #     self._logger.debug(f'tick: {tick}')
        #     for src, item in order.items():
        #         for dest, qty in item.items():
        #             self._logger.debug(f'  {src} -> {dest}: {qty}')
        #             if tick > 0 and total_needs[tick][src] == 0:
        #                 total_needs[tick][src] += total_needs[tick - 1][src] + qty
        #             else:
        #                 total_needs[tick][src] += qty
        # for tick in range(configs.params.window_size):
        #     self._logger.debug(f'Accumulated Needs {tick}: {total_needs[tick]}')

        # self._logger.debug('\n*********************** Vessel Arrival ***********************')
        # for tick, info in tick_vessel_port_connection.items():
        #     self._logger.debug('tick:', tick)
        #     for vessel, port in info.items():
        #         self._logger.debug(f'  [{vessel}] in [{port}]')

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

        # Initialize decision variables list
        # corresponding to order
        self._order_apply = [{} for i in range(self._window_size)]      # [s][p1][p2]
        self._port_full = [{} for i in range(self._window_size)]        # [s][p1][p2]
        self._full_has_loaded = [{} for i in range(self._window_size)]
        
        # corresponding to valid (port, vessel)
        self._load_full_all = [{} for i in range(self._window_size)]  # [s][p1][p2][v]
        self._load_full_delayed = [{} for i in range(self._window_size)]   # [s][p1][p2][v]
        self._load_full = [{} for i in range(self._window_size)]        # [s][p1][p2][v]
        self._discharge_full = [{} for i in range(self._window_size)]   # [s][p2][v]
        
        self._load_empty = [{} for i in range(self._window_size)]       # [s][p][v]
        self._discharge_empty = [{} for i in range(self._window_size)]  # [s][p][v]
        
        # corresponding to time
        self._port_empty = [{} for i in range(self._window_size)]       # [s][p], empty inventory
        self._port_on_consignee = [{} for i in range(self._window_size)]    # [s][p], on_consignee increment
        
        self._vessel_empty = [{} for i in range(self._window_size)] # [s][v], empty inventory
        self._vessel_full = [{} for i in range(self._window_size)] # [s][v][p2], full inventory

        # Initialize decision variables related to port & vessel keys
        for s in range(self._window_size):
            for p in self._ports:
                self._order_apply[s][p] = dict()
                self._port_full[s][p] = dict()
                self._full_has_loaded[s][p] = dict()
                
                self._load_full_all[s][p] = dict()
                self._load_full_delayed[s][p] = dict()
                self._load_full[s][p] = dict()
                self._discharge_full[s][p] = dict()
                
                self._load_empty[s][p] = dict()
                self._discharge_empty[s][p] = dict()

                for p2 in self._ports:
                    self._load_full_all[s][p][p2] = dict()
                    self._load_full_delayed[s][p][p2] = dict()
                    self._load_full[s][p][p2] = dict()

            for v in self._vessels:
                self._vessel_full[s][v] = dict()

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

    def _get_safety_inventory(self):
        self._safety_inventory = dict()
        for s in range(self._window_size):
            self._safety_inventory[s] = dict()
            for p in self._ports:
                self._safety_inventory[s][p] = 0

    def _find_next_coming_vessel(self, order_step, src_port, dest_port):
        reachable_vessel_list = [vessel for vessel in self._vessels if src_port in self._route_dict[vessel] and dest_port in self._route_dict[vessel]]
        
        for step in range(order_step + 1, self._window_size):
            for vessel in reachable_vessel_list:
                if self._step_vessel_port_connection[step][vessel] == src_port:
                    return step, vessel
        
        return None, None

    def _find_previous_comming_vessel(self, cur_step, src_port, dest_port):
        reachable_vessel_list = [vessel for vessel in self._vessels if src_port in self._route_dict[vessel] and dest_port in self._route_dict[vessel]]
        for step in (cur_step - 1, -1, -1):
            for vessel in reachable_vessel_list:
                if self._step_vessel_port_connection[step][vessel] == src_port:
                    return step, vessel
        return None, None

    # Decision Variables
    def _init_variables(self,
                        initial_port_empty: dict,
                        initial_port_on_shipper: dict,
                        initial_port_on_consignee: dict,
                        initial_port_full: dict,
                        initial_vessel_empty: dict,
                        initial_vessel_full: dict
                        ):
        for s in range(self._window_size):
            for p in self._ports:
                self._port_empty[s][p] = 0
                self._port_on_consignee[s][p] = 0
                
                for v in self._vessels:
                    self._discharge_full[s][p][v] = 0
                    self._load_empty[s][p][v] = 0
                    self._discharge_empty[s][p][v] = 0
                
                for p2 in self._ports:
                    self._order_apply[s][p][p2] = 0
                    self._port_full[s][p][p2] = 0
                    self._full_has_loaded[s][p][p2] = True

                    for v in self._vessels:
                        self._load_full_all[s][p][p2][v] = 0
                        self._load_full_delayed[s][p][p2][v] = 0
                        self._load_full[s][p][p2][v] = 0
                   
            for v in self._vessels:
                self._vessel_empty[s][v] = 0
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
                                self._load_full_all[vessel_arrive_time][p1][p2][vessel_code] = \
                                    LpVariable(name=f'Load_Full_ALL_{vessel_arrive_time}_{p1}_{p2}_{vessel_code}', lowBound=0, cat='Integer')
                                self._load_full_delayed[vessel_arrive_time][p1][p2][vessel_code] = \
                                    LpVariable(name=f'Load_Full_DELAYED_{vessel_arrive_time}_{p1}_{p2}_{vessel_code}', lowBound=0, cat='Integer')

    def _load_values(self, path):
        for line in open(path, 'r').readlines():
            if '[' not in line:
                continue
            # try:
            parts = re.findall(r'\[(.*?)\]', line)
            key = parts[0]
            tick = int(parts[1])
            s = tick - self._global_tick
            if key in ['order_apply', 'port_full', 'Full Return']:
                p1 = parts[2]
                p2 = parts[3]
                qty = int(parts[4])
                if key == 'order_apply':
                    self._order_apply[s][p1][p2] = qty
                elif key == 'Full Return':
                    self._port_full[s][p1][p2] = qty
                # elif key == 'port_full':
                #     self._port_full[s][p1][p2] = qty
            elif key in ['port_empty', 'port_on_consignee']:
                p = parts[2]
                qty = int(parts[3])
                if key == 'port_empty':
                    self._port_empty[s][p] = qty
                elif key == 'port_on_consignee':
                    assert s > 0
                    self._port_on_consignee[s - 1][p] = qty
            elif key == 'vessel_empty':
                v = parts[2]
                qty = int(parts[3])
                self._vessel_empty[s][v] = qty
            elif key == 'vessel_full':
                v = parts[2]
                p2 = parts[3]
                qty = int(parts[4])
                self._vessel_full[s][v][p2] = qty
            elif key == 'load_full':
                p1 = parts[2]
                p2 = parts[3]
                v = parts[4]
                qty = int(parts[5])
                self._load_full[s][p1][p2][v] = qty
            elif key == 'discharge_full':
                p2 = parts[2]
                v = parts[3]
                qty = int(parts[4])
                if isinstance(self._discharge_full[s][p2][v], int):
                    self._discharge_full[s][p2][v] += qty
                else:
                    self._discharge_full[s][p2][v] = qty
            elif key in ['load_empty', 'discharge_empty']:
                p = parts[2]
                v = parts[3]
                qty = int(parts[4])
                if key == 'load_empty':
                    self._load_empty[s][p][v] = qty
                elif key == 'discharge_empty':
                    self._discharge_empty[s][p][v] = qty

        for s in range(self._window_size):
            for p in self._ports:
                for v in self._vessels:
                    if isinstance(self._load_empty[s][p][v], int) and isinstance(self._discharge_empty[s][p][v], int):
                        assert self._load_empty[s][p][v] == 0 or self._discharge_empty[s][p][v] == 0
                    elif isinstance(self._load_empty[s][p][v], int):
                        self._discharge_empty[s][p][v] = 0
                    elif isinstance(self._discharge_empty[s][p][v], int):
                        self._load_empty[s][p][v] = 0
                    else:
                        self._action_check_logger.debug(f'load_empty[{s}][{p}][{v}] = {self._load_empty[s][p][v]} | discharge_empty[{s}][{p}][{v}] = {self._discharge_empty[s][p][v]}')

    def _set_objective(self, problem):
        if self._enable_time_decay:
            order_gain = self._order_gain_factor * lpSum([math.pow(self._time_decay, s) * self._order_apply[s][p1][p2] \
                for s in range(self._window_size) \
                    for p1 in self._order_apply[s].keys() \
                        for p2 in self._order_apply[s][p1].keys()])
        else:
            order_gain = self._order_gain_factor * lpSum([self._order_apply[s][p1][p2] \
                for s in range(self._window_size) \
                    for p1 in self._order_apply[s].keys() \
                        for p2 in self._order_apply[s][p1].keys()])

        transit_cost = self._transit_cost_factor * lpSum([self._vessel_empty[s][v] \
            for s in range(len(self._step_vessel_port_connection)) \
                for v in self._vessels \
                    if self._step_vessel_port_connection[s][v] == None])

        load_discharge_cost = self._load_discharge_cost_factor * lpSum([self._load_empty[s][p][v] + self._discharge_empty[s][p][v] \
            for s in range(len(self._step_vessel_port_connection)) \
                for v, p in self._step_vessel_port_connection[s].items() \
                    if p != None])

        full_delayed_punishment = self._full_delayed_factor * lpSum([self._load_full_delayed[s][p1][p2][v] \
            for s in range(self._window_size) \
                for p1 in self._ports \
                    for p2 in self._ports \
                        for v in self._vessels])

        # TODO: add soft safety_inventory
        problem += order_gain - transit_cost - load_discharge_cost - full_delayed_punishment

    def _add_constrants(self, problem):
        """ Processing Order of Events [2019-12-12]:
        BE.step() => EVENT_BUFFER.execute(current_tick)
        
        VESSEL_DEPARTURE => EMPTY_RETURN => FULL_RETURN => ORDER => VESSEL_ARRIVAL(Load_Full, Early_Discharge) => DECISION_EVENT
                    DISCHARGE_FULL      => FULL_RETURN => ORDER => VESSEL_ARRIVAL                              => DECISION_EVENT
        """
        # Order processing
        # TODO: multi orders coming in the same tick???
        for s in self._orders.keys():
            for p1 in self._orders[s].keys():
                for p2 in self._orders[s][p1].keys():
                    # Order apply: operation constraint 1/2
                    problem += self._order_apply[s][p1][p2] <= self._orders[s][p1][p2]
                    if self._load_action and self._enable_loaded_assert:
                        assert isinstance(self._order_apply[s][p1][p2], int)
                        assert self._order_apply[s][p1][p2] <= self._orders[s][p1][p2]

                    # Port Full:
                    s_lo = s + self._full_return_buffer_ticks[p1]
                    if s_lo < self._window_size:
                        problem += self._order_apply[s][p1][p2] == self._port_full[s_lo][p1][p2]
                        if self._load_action and self._enable_loaded_assert:
                            if isinstance(self._port_full[s_lo][p1][p2], int):
                                assert self._order_apply[s][p1][p2] == self._port_full[s_lo][p1][p2]
                            else:
                                self._action_check_logger.debug(f'order_apply[{s}][{p1}][{p2}] = {self._order_apply[s][p1][p2]} | port_full[{s_lo}][{p1}][{p2}] = {self._port_full[s_lo][p1][p2]}')

        # Load Full: operation constraint
        # [2019-12-12] For current implementation, FULL_RETURN of vessel_arrive_time can be loaded in vessel_arrive_time
        for s in self._orders.keys():
            for p1 in self._orders[s].keys():
                for p2 in self._orders[s][p1].keys():
                    s_lo = s + self._full_return_buffer_ticks[p1]
                    vessel_arrive_time, vessel_code = self._find_next_coming_vessel(s_lo, p1, p2)
                    # start_time = s_lo if s == 0 else s_lo + 1
                    start_time = 0 if s == 0 else s_lo + 1
                    if not vessel_arrive_time or not vessel_arrive_time < self._window_size or self._full_has_loaded[start_time][p1][p2]:
                        continue
                    for time_before_arrive in range(start_time, vessel_arrive_time + 1):
                        self._full_has_loaded[time_before_arrive][p1][p2] = True
                    problem += self._load_full_all[vessel_arrive_time][p1][p2][vessel_code] == \
                        lpSum([self._port_full[si][p1][p2] for si in range(start_time, vessel_arrive_time + 1)])
                    problem += self._load_full_delayed[vessel_arrive_time][p1][p2][vessel_code] <= \
                        self._load_full_all[vessel_arrive_time][p1][p2][vessel_code]
                    previous_time, previous_vessel = self._find_previous_comming_vessel(vessel_arrive_time, p1, p2)
                    if (previous_time is not None) and (previous_vessel is not None):
                        problem += self._load_full[vessel_arrive_time][p1][p2][vessel_code] == \
                            self._load_full_all[vessel_arrive_time][p1][p2][vessel_code] \
                                - self._load_full_delayed[vessel_arrive_time][p1][p2][vessel_code] \
                                    + self._load_full_delayed[previous_time][p1][p2][previous_vessel]
                    else:
                        problem += self._load_full[vessel_arrive_time][p1][p2][vessel_code] == \
                            self._load_full_all[vessel_arrive_time][p1][p2][vessel_code] \
                                - self._load_full_delayed[vessel_arrive_time][p1][p2][vessel_code]

                    if self._load_action and self._enable_loaded_assert:
                        full_list = [self._port_full[si][p1][p2] for si in range(start_time, vessel_arrive_time + 1)]
                        if not isinstance(self._load_full[vessel_arrive_time][p1][p2][vessel_code], int):
                            self._action_check_logger.debug(f'load_full[{vessel_arrive_time}][{p1}][{p2}][{vessel_code}] is variable | port_full list {s} [{s_lo}, {vessel_arrive_time}] | | {full_list}')
                        else:
                            sum_num = sum(full_list)
                            if not self._load_full[vessel_arrive_time][p1][p2][vessel_code] == sum_num:
                                self._action_check_logger.debug(f'load_full[{vessel_arrive_time}][{p1}][{p2}][{vessel_code}] = {self._load_full[vessel_arrive_time][p1][p2][vessel_code]} | port_full list {s} [{s_lo}, {vessel_arrive_time}] | | {sum_num} | {full_list}')

        # Discharge Full: operation constraint
        for s in range(self._window_size):
            for v, p in self._step_vessel_port_connection[s].items():
                if p != None:
                    problem += self._discharge_full[s][p][v] == self._vessel_full[s][v][p]
                    if self._load_action and self._enable_loaded_assert:
                        if isinstance(self._discharge_full[s][p][v], int) and isinstance(self._vessel_full[s][v][p], int):
                            assert self._discharge_full[s][p][v] == self._vessel_full[s][v][p]
                        elif isinstance(self._discharge_full[s][p][v], int):
                            assert self._discharge_full[s][p][v] == 0
                        elif isinstance(self._vessel_full[s][v][p], int):
                            assert self._vessel_full[s][v][p] == 0
                        else:
                            self._action_check_logger.debug(f'discharge_full[{s}][{p}][{v}] = {self._discharge_full[s][p][v]} | vessel_full[{s}][{v}][{p}] = {self._vessel_full[s][v][p]}')

        # Port On-Consignee:
        for s in range(self._window_size):
            for p in self._ports:
                problem += self._port_on_consignee[s][p] == lpSum([self._discharge_full[s][p][v] for v in self._vessels])
                if self._load_action and self._enable_loaded_assert:
                    if not isinstance(self._port_on_consignee[s][p], int):
                        self._action_check_logger.debug(f'port_on_consignee[{s}][{p}] is variable | discharge_full_list: {discharge_full_list} ')
                    else:
                        discharge_full_list = [self._discharge_full[s][p][v] for v in self._vessels]
                        sum_num = sum(discharge_full_list)
                        if not self._port_on_consignee[s][p] == sum_num:
                            self._action_check_logger.debug(f'port_on_consignee[{s}][{p}] = {self._port_on_consignee[s][p]} | discharge_full_list: {discharge_full_list} | {sum_num}')
        
        # Order apply: operation constraint 2/2
        # Load empty: operation constraint
        """ Usage of Empty [2019-12-12]:
            Empty discharged in current tick can be loaded to another vessel, in the order of BE._vessels
        """
        for s in range(self._window_size):
            for p in self._ports:
                s_li = s - self._empty_return_buffer_ticks[p]
                if s_li >= 0:
                    for vessel_idx in range(len(self._vessels)):
                        problem += self._port_empty[s][p] + self._port_on_consignee[s_li][p] - self._safety_inventory[s][p] \
                            - lpSum([self._order_apply[s][p][des_p] for des_p in self._ports]) \
                                + lpSum([self._discharge_empty[s][p][self._vessels[idx]] for idx in range(vessel_idx)]) \
                                    >= lpSum([self._load_empty[s][p][self._vessels[idx]] for idx in range(vessel_idx + 1)]) 
                else:
                    for vessel_idx in range(len(self._vessels)):
                        problem += self._port_empty[s][p] - self._safety_inventory[s][p] \
                            - lpSum([self._order_apply[s][p][des_p] for des_p in self._ports]) \
                                + lpSum([self._discharge_empty[s][p][self._vessels[idx]] for idx in range(vessel_idx)]) \
                                    >= lpSum([self._load_empty[s][p][self._vessels[idx]] for idx in range(vessel_idx + 1)]) 

                if self._load_action and self._enable_loaded_assert:
                    assert isinstance(self._port_empty[s][p], int)
                    assert isinstance(self._safety_inventory[s][p], int)
                    for des_p in self._ports:
                        assert isinstance(self._order_apply[s][p][des_p], int)
                    has_violation = False
                    for v in self._vessels:
                        if not isinstance(self._load_empty[s][p][v], int):
                            self._action_check_logger.debug(f'load_empty[{s}][{p}][{v}] is variable')
                            has_violation = True
                    if has_violation:
                        continue
                    order_apply_list = [self._order_apply[s][p][des_p] for des_p in self._ports]
                    load_empty_list = [self._load_empty[s][p][v] for v in self._vessels]
                    discharge_empty_list = [self._discharge_empty[s][p][v] for v in self._vessels]
                    if s_li >= 0:
                        if not isinstance(self._port_on_consignee[s_li][p], int):
                            self._action_check_logger.debug(f'port_on_consignee[{s_li}][{p}] is variable')
                        else:
                            if not self._port_empty[s][p] + self._port_on_consignee[s_li][p] - self._safety_inventory[s][p] + sum(discharge_empty_list) >= sum(order_apply_list) + sum(load_empty_list):
                                self._action_check_logger.debug(f'port_empty[{s}][{p}] = {self._port_empty[s][p]} | port_on_consignee[{s_li}][{p}] = {self._port_on_consignee[s_li][p]} | order_apply_list: {sum(order_apply_list)} | {order_apply_list} || load_empty_list: {sum(load_empty_list)} | {load_empty_list}')
                    else:
                        if not self._port_empty[s][p] - self._safety_inventory[s][p] + sum(discharge_empty_list) >= sum(order_apply_list) + sum(load_empty_list):
                            self._action_check_logger.debug(f'port_empty[{s}][{p}] = {self._port_empty[s][p]} | order_apply_list: {sum(order_apply_list)} | {order_apply_list} || load_empty_list: {sum(load_empty_list)} | {load_empty_list}')


        # Discharge Empty: operation constraint
        for s in range(self._window_size):
            for v, p in self._step_vessel_port_connection[s].items():
                if p != None:
                    problem += self._discharge_empty[s][p][v] <= self._vessel_empty[s][v]
                    if self._load_action and self._enable_loaded_assert:
                        if isinstance(self._discharge_empty[s][p][v], int) and isinstance(self._vessel_empty[s][v], int):
                            assert self._discharge_empty[s][p][v] <= self._vessel_empty[s][v]
                        else:
                            self._action_check_logger.debug(f'vessel_empty[{s}][{v}] = {self._vessel_empty[s][v]} | discharge_empty[{s}][{p}][{v}] = {self._discharge_empty[s][p][v]}')

        # Inventory of vessels
        for s in range(1, self._window_size):
            for v in self._vessels:
                pre_p = self._step_vessel_port_connection[s-1][v]
                # Empty: recursion formula
                if pre_p != None:
                    problem += self._vessel_empty[s][v] == self._vessel_empty[s-1][v] \
                        + self._load_empty[s-1][pre_p][v] - self._discharge_empty[s-1][pre_p][v]
                    if self._load_action and self._enable_loaded_assert:
                        if isinstance(self._vessel_empty[s][v], int) and isinstance(self._vessel_empty[s-1][v], int) \
                            and isinstance(self._load_empty[s-1][pre_p][v], int) and isinstance(self._discharge_empty[s-1][pre_p][v], int):
                                assert self._vessel_empty[s][v] == self._vessel_empty[s-1][v] + self._load_empty[s-1][pre_p][v] - self._discharge_empty[s-1][pre_p][v]
                        else:
                            self._action_check_logger.debug(f'vessel empty: {self._vessel_empty[s][v]} | {self._vessel_empty[s-1][v]} | {self._load_empty[s-1][pre_p][v]} | {self._discharge_empty[s-1][pre_p][v]}')
                else:
                    problem += self._vessel_empty[s][v] == self._vessel_empty[s-1][v]
                    if self._load_action and self._enable_loaded_assert:
                        if isinstance(self._vessel_empty[s][v], int) and isinstance(self._vessel_empty[s-1][v], int):
                            assert self._vessel_empty[s][v] == self._vessel_empty[s-1][v]
                        else:
                            self._action_check_logger.debug(f'vessel_empty[{s}][{v}] = {self._vessel_empty[s][v]} | vessel_empty[{s-1}][{v}] = {self._vessel_empty[s-1][v]}')
                # Full: recursion formula
                for p2 in self._ports:
                    if pre_p != None:
                        problem += self._vessel_full[s][v][p2] == self._vessel_full[s-1][v][p2] \
                            + self._load_full[s-1][pre_p][p2][v] - self._discharge_full[s-1][p2][v]
                        if self._load_action and self._enable_loaded_assert:
                            if isinstance(self._vessel_full[s][v][p2], int) and isinstance(self._vessel_full[s-1][v][p2], int) \
                                and isinstance(self._load_full[s-1][pre_p][p2][v], int) and isinstance(self._discharge_full[s-1][p2][v], int):
                                    assert self._vessel_full[s][v][p2] == self._vessel_full[s-1][v][p2] + self._load_full[s-1][pre_p][p2][v] - self._discharge_full[s-1][p2][v]
                            else:
                                self._action_check_logger.debug(f'vessel full: {self._vessel_full[s][v][p2]} | {self._vessel_full[s-1][v][p2]} | {self._load_full[s-1][pre_p][p2][v]} | {self._discharge_full[s-1][p2][v]}')
                    else:
                        problem += self._vessel_full[s][v][p2] == self._vessel_full[s-1][v][p2]
                        if self._load_action and self._enable_loaded_assert:
                            if isinstance(self._vessel_full[s][v][p2], int) and isinstance(self._vessel_full[s-1][v][p2], int):
                                assert self._vessel_full[s][v][p2] == self._vessel_full[s-1][v][p2]
                            else:
                                self._action_check_logger.debug(f'vessel_full[{s}][{v}][{p2}] = {self._vessel_full[s-1][v][p2]} | vessel_full[{s-1}][{v}][{p2}] = {self._vessel_full[s-1][v][p2]}')
        
        # Inventory of ports
        for s in range(1, self._window_size):
            for p in self._ports:
                # Empty: recursion formula
                s_li = s - 1 - self._empty_return_buffer_ticks[p]
                if self._load_action and self._enable_loaded_assert:
                    assert isinstance(self._port_empty[s][p], int)
                    assert isinstance(self._port_empty[s-1][p], int)
                    has_violation = False
                    for v in self._vessels:
                        if not isinstance(self._discharge_empty[s-1][p][v], int):
                            self._action_check_logger.debug(f'discharge_empty[{s-1}][{p}][{v}] is variable')
                        if not isinstance(self._load_empty[s-1][p][v], int):
                            self._action_check_logger.debug(f'load_empty[{s-1}][{p}][{v}] is variable')
                    for des_p in self._ports:
                        assert isinstance(self._order_apply[s-1][p][des_p], int)
                    load_discharge_empty_list = [self._discharge_empty[s-1][p][v] - self._load_empty[s-1][p][v] for v in self._vessels]
                    order_apply_list = [self._order_apply[s-1][p][des_p] for des_p in self._ports]
                    load_discharge_empty_sum = sum(load_discharge_empty_list)
                    order_apply_sum = sum(order_apply_list)
                if s_li >= 0:
                    problem += self._port_empty[s][p] == self._port_empty[s-1][p] \
                        + lpSum([self._discharge_empty[s-1][p][v] - self._load_empty[s-1][p][v] for v in self._vessels]) \
                            - lpSum([self._order_apply[s-1][p][des_p] for des_p in self._ports]) \
                                + self._port_on_consignee[s_li][p]
                    if self._load_action and self._enable_loaded_assert:
                        if not isinstance(self._port_on_consignee[s_li][p], int):
                            self._action_check_logger.debug(f'port_on_consignee[{s_li}][{p}] is variable')
                        else:
                            if not self._port_empty[s][p] == self._port_empty[s-1][p] + load_discharge_empty_sum - order_apply_sum + self._port_on_consignee[s_li][p]:
                                self._action_check_logger.debug(f'port_empty: {self._port_empty[s][p]} | {self._port_empty[s-1][p]} | {load_discharge_empty_sum} | {order_apply_sum} | {self._port_on_consignee[s_li][p]} | {load_discharge_empty_list} | {order_apply_list}')
                else:
                    problem += self._port_empty[s][p] == self._port_empty[s-1][p] \
                        + lpSum([self._discharge_empty[s-1][p][v] - self._load_empty[s-1][p][v] for v in self._vessels]) \
                            - lpSum([self._order_apply[s-1][p][des_p] for des_p in self._ports])
                    if self._load_action and self._enable_loaded_assert:
                        if not self._port_empty[s][p] == self._port_empty[s-1][p] + load_discharge_empty_sum - order_apply_sum:
                            self._action_check_logger.debug(f'port_empty: {self._port_empty[s][p]} | {self._port_empty[s-1][p]} | {load_discharge_empty_sum} | {order_apply_sum} | {load_discharge_empty_list} | {order_apply_list}')

        # Capacity constraint
        for s in range(self._window_size):
            # Port TODO: consider shipper & consignee
            for p in self._ports:
                problem += self._port_capacity[p] >= self._port_empty[s][p]
                if self._load_action and self._enable_loaded_assert:
                    assert isinstance(self._port_capacity[p], int) or isinstance(self._port_capacity[p], float)
                    assert isinstance(self._port_empty[s][p], int)
                    assert self._port_capacity[p] >= self._port_empty[s][p]
            # Vessel
            for v in self._vessels:
                problem += self._vessel_capacity[v] >= self._vessel_empty[s][v] \
                    + lpSum([self._vessel_full[s][v][des_p] for des_p in self._ports])
                if self._load_action and self._enable_loaded_assert:
                    assert isinstance(self._vessel_capacity[v], int) or isinstance(self._vessel_capacity[v], float)
                    assert isinstance(self._vessel_empty[s][v], int)
                    has_violation = False
                    for des_p in self._ports:
                        if not isinstance(self._vessel_full[s][v][des_p], int):
                            self._action_check_logger.debug(f'vessel_full[{s}][{v}][{des_p}] is variable')
                            has_violation = True
                    if not has_violation:
                        assert self._vessel_capacity[v] >= self._vessel_empty[s][v] + sum([self._vessel_full[s][v][des_p] for des_p in self._ports])

    def _shortage_calculation(self):
        self._shortage = [0] * self._window_size
        self._total_shortage = 0
        for step in self._orders.keys():
            self._shortage[step] = 0
            for p1 in self._orders[step].keys():
                for port2 in self._orders[step][p1].keys():
                    self._shortage[step] += self._orders[step][p1][port2] - get_num(self._order_apply[step][p1][port2])
            self._total_shortage += self._shortage[step]

    def _write_solution_to_file(self):        
        solution_logger = Logger(tag=f'{self._file_prefix}_{self._global_tick}', format_=LogFormat.none,
            dump_folder=self._log_folder, dump_mode='w', auto_timestamp=False, extension_name='txt')

        for step in range(self._window_size):
            solution_logger.debug(f'******************** Tick {self._global_tick + step} / Step {step} ********************')
            for port in self._ports:
                solution_logger.debug(f'[{port}]')
                solution_logger.debug(f'    Empty Inventory: {get_num(self._port_empty[step][port])}')
                solution_logger.debug(f'    On-Consignee Increment: {get_num(self._port_on_consignee[step][port])}')
                # if port in self._port_full[step].keys():
                #     for p2 in self._port_full[step][port].keys():
                #         solution_logger.debug(f'    Full for {p2}: {get_num(self._port_full[step][port][p2])}')
                if step in self._orders.keys() and port in self._orders[step].keys():
                    for p2 in self._orders[step][port].keys():
                        solution_logger.debug(f'    Order Apply to {p2}: {get_num(self._order_apply[step][port][p2])} / {self._orders[step][port][p2]}')
                for vessel in self._vessels:
                    if self._step_vessel_port_connection[step][vessel] == port:
                        solution_logger.debug(f'    [{vessel}]')
                        solution_logger.debug(f'        Empty Inventory: {get_num(self._vessel_empty[step][vessel])}')
                        for p2 in self._ports:
                            # if get_num(self._load_full[step][port][p2][vessel]) > 0:
                                # solution_logger.debug(f'        Load Full for {p2}: {get_num(self._load_full[step][port][p2][vessel])}')
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
                            initial_port_on_shipper: dict,
                            initial_port_on_consignee: dict,
                            initial_port_full: dict,
                            initial_vessel_empty: dict,
                            initial_vessel_full: dict,
                            tick_orders: dict = None,
                            tick_vessel_port_connection: dict = None
                            ):
        self._global_tick = current_tick
        self._set_step_orders(current_tick=current_tick, tick_orders=tick_orders)
        self._set_step_vessel_port_connection(current_tick=current_tick, tick_vessel_port_connection=tick_vessel_port_connection)
        self._get_safety_inventory()

        problem = LpProblem(name=f"ecr Problem: from Tick_{current_tick}", sense=LpMaximize)
        self._init_variables(initial_port_empty=initial_port_empty,
                             initial_port_on_shipper=initial_port_on_shipper,
                             initial_port_on_consignee=initial_port_on_consignee,
                             initial_port_full=initial_port_full,
                             initial_vessel_empty=initial_vessel_empty,
                             initial_vessel_full=initial_vessel_full
                             )
        if self._load_action:
            self._load_values(self._action_path)
        self._set_objective(problem)
        self._add_constrants(problem)
        # Default: using CBC solver
        problem.solve()
        # problem.solve(GLPK())

        lp_file_path = os.path.join(self._log_folder, f'{self._file_prefix}_{self._global_tick}.lp')
        problem.writeLP(lp_file_path)

        if self._load_action:
            if problem.status != 1:
                print(f'==================== NOT OPTIMAL SOLUTION FOR LP FORMULATION ====================')
        else:
            assert (problem.status == 1)

        # Update the end point of apply buffer
        self._apply_buffer_end = current_tick + self._apply_buffer_size
        self._applied_idx = {vessel: -1 for vessel in self._vessels}

        self._solution_status = LpStatus[problem.status]
        self._objective_gotten = value(problem.objective)
        self._shortage_calculation()

        # Show the details of the lp solution
        self._write_solution_to_file()

    def _find_next_decision_index(self, port_code: str, vessel_code: str):
        self._applied_idx[vessel_code] += 1
        while self._applied_idx[vessel_code] < self._window_size:
            if self._step_vessel_port_connection[self._applied_idx[vessel_code]][vessel_code] == port_code:
                break
            self._applied_idx[vessel_code] += 1
        return self._applied_idx[vessel_code]

    def choose_action(self,
                      current_tick: int,
                      port_code: str,
                      vessel_code: str,
                      initial_port_empty: dict = None,
                      initial_port_on_shipper: dict = None,
                      initial_port_on_consignee: dict = None,
                      initial_port_full: dict = None,
                      initial_vessel_empty: dict = None,
                      initial_vessel_full: dict = None,
                      tick_orders: dict = None,
                      tick_vessel_port_connection: dict = None
                      ):
        assert current_tick < self._apply_buffer_end
        # if current_tick >= self._apply_buffer_end:
        #     self.formulate_and_solve(current_tick=current_tick,
        #                              initial_port_empty=initial_port_empty,
        #                              initial_port_on_shipper=initial_port_on_shipper,
        #                              initial_port_on_consignee=initial_port_on_consignee,
        #                              initial_port_full=initial_port_full,
        #                              initial_vessel_empty=initial_vessel_empty,
        #                              initial_vessel_full=initial_vessel_full,
        #                              tick_orders=tick_orders,
        #                              tick_vessel_port_connection=tick_vessel_port_connection
        #                              )

        if self._configs.enable_arrival_noise:
            step = self._find_next_decision_index(port_code=port_code, vessel_code=vessel_code)
            if step >= self._window_size:
                return 0
        else:
            step = current_tick - self._global_tick

        num_load_empty = get_num(self._load_empty[step][port_code][vessel_code])
        num_discharge_empty = get_num(self._discharge_empty[step][port_code][vessel_code])

        # Execute Action
        return num_discharge_empty - num_load_empty
    
    def clear(self):
        self._apply_buffer_end = 0
        self._applied_idx.clear()
        self._tick_orders.clear()
        self._tick_vessel_port_connection.clear()