from pulp import LpProblem, LpVariable, LpMaximize, lpSum

import math
import numpy as np

from maro.simulator.scenarios.ecr.common import EcrEventType

INVALID_TICK = -1

class Forecast():
    def __init__(self, 
                 moving_average_length: int, 
                 port_idx2name: dict, 
                 vessel_idx2name: dict, 
                 port_name_list: dict, 
                 vessel_name_list: dict(), 
                 topo_config: dict):
        self._moving_average_length = moving_average_length
        self._port_idx2name = port_idx2name
        self._vessel_idx2name = vessel_idx2name
        self._port_name_list = port_name_list
        self._vessel_name_list = vessel_name_list
        self._topo_config = topo_config

        self._port_idx_list = []
        self._vessel_idx_list = []
        for idx in port_idx2name.keys():
            self._port_idx_list.append(idx)
        for idx in vessel_idx2name.keys():
            self._vessel_idx_list.append(idx)
        
        self._vessel_cycle_time, self._vessel_in_cycle_arrive_step = self._calculate_vessel_related_time()
        
        self._order_history = dict() # [port][tick]
        self._return_empty_history = dict() # [port][tick]
        self._full_delta_history = dict()   # [vessel][port][tick]
        self._vessel_arrival_history = dict()  # [tick][vessel][port]

        self._last_order_store_tick = 0
        self._last_return_empty_store_tick = 0
        self._last_full_delta_store_tick = 0
        self._last_v2p_store_tick = 0

    def augment_history_record(self, current_tick, finished_events, snapshot_list):
        last_order_tick = INVALID_TICK
        last_return_empty_tick = INVALID_TICK
        last_v2p_tick = INVALID_TICK
        for event in finished_events:
            if event.event_type == EcrEventType.ORDER:# TODO: event tick should be unified
                if event.payload.tick > self._last_order_store_tick:
                    if event.payload.tick > last_order_tick:
                        last_order_tick = event.payload.tick 
                    tick = event.payload.tick
                    port = self._port_idx2name[event.payload.src_port_idx]
                    qty = event.payload.quantity
                    self._order_history.setdefault(port, dict())
                    self._order_history[port].setdefault(tick, 0)
                    self._order_history[port][tick] += qty
            elif event.event_type == EcrEventType.RETURN_EMPTY:
                if event.tick > self._last_return_empty_store_tick:
                    if event.tick > last_return_empty_tick:
                        last_return_empty_tick = event.tick
                    tick = event.tick
                    port = self._port_idx2name[event.payload[1]]
                    qty = event.payload[0]
                    self._return_empty_history.setdefault(port, dict())
                    self._return_empty_history[port].setdefault(tick, 0)
                    self._return_empty_history[port][tick] += qty
            elif event.event_type == EcrEventType.LOAD_FULL:
                if event.tick > self._last_v2p_store_tick:
                    if event.tick > last_v2p_tick:
                        last_v2p_tick = event.tick
                    tick = event.tick
                    port_name = self._port_idx2name[event.payload.port_idx]
                    vessel_name = self._vessel_idx2name[event.payload.vessel_idx]
                    self._vessel_arrival_history.setdefault(tick, dict())
                    self._vessel_arrival_history[tick][vessel_name] = port_name

        if last_order_tick != INVALID_TICK:
            self._last_order_store_tick = last_order_tick
        if last_return_empty_tick != INVALID_TICK:
            self._last_return_empty_store_tick = last_return_empty_tick
        if last_v2p_tick != INVALID_TICK:
            self._last_v2p_store_tick = last_v2p_tick

        #need to have an event on load full        
        pre_full_on_vessels = []
        for tick in range(self._last_full_delta_store_tick, current_tick):
            full_on_vessels = snapshot_list.matrix[tick: 'full_on_vessels'].reshape(len(self._vessel_name_list), len(self._port_name_list))
            if pre_full_on_vessels != []:
                for vessel_idx in self._vessel_idx_list:
                    vessel_name = self._vessel_idx2name[vessel_idx]
                    if tick in self._vessel_arrival_history.keys() and vessel_name in self._vessel_arrival_history[tick].keys():
                        port_name = self._vessel_arrival_history[tick][vessel_name]
                        self._full_delta_history.setdefault(vessel_name, dict())
                        self._full_delta_history[vessel_name].setdefault(port_name, dict())
                        self._full_delta_history[vessel_name][port_name][tick] = np.sum(full_on_vessels[vessel_idx]) - np.sum(pre_full_on_vessels[vessel_idx])
            pre_full_on_vessels = full_on_vessels
        self._last_full_delta_store_tick = current_tick - 1
    
    def _calculate_vessel_related_time(self):
        vessel_in_cycle_arrive_step = dict()
        vessel_cycle_time = dict()

        for vessel_name, vessel_info in self._topo_config['vessels'].items():
            vessel_route_name = vessel_info['route']['route_name']
            vessel_speed = vessel_info['sailing']['speed']
            vessel_duration = vessel_info['parking']['duration']
            vessel_route = self._topo_config['routes'][vessel_route_name]

            vessel_in_cycle_arrive_step.setdefault(vessel_name, dict())
            vessel_cycle_time[vessel_name] = 0
            # calculate cycle time
            for port in vessel_route:
                vessel_cycle_time[vessel_name] += round(port['distance'] / vessel_speed) + vessel_duration

                vessel_in_cycle_arrive_step[vessel_name][vessel_cycle_time[vessel_name] - vessel_duration] = port['port_name'] 

        return vessel_cycle_time, vessel_in_cycle_arrive_step

    def calculate_vessel_arrival(self, topo_config, snapshot_list, current_tick, forecast_length):
        step_vessel_port_connection = dict()
        vessel_plans = snapshot_list.matrix[current_tick : "vessel_plans"].reshape(len(self._vessel_idx_list), len(self._port_idx_list))
        # print(vessel_plans)
        
        # calculate vessel next stop port
        vessel_next_stop = dict()
        for vessel_idx in self._vessel_idx_list:
            for port_idx in self._port_idx_list:
                if vessel_plans[vessel_idx][port_idx] == -1:
                    vessel_plans[vessel_idx][port_idx] = 1000000  #TODO: change to max int
            vessel_next_stop[self._vessel_idx2name[vessel_idx]] = (self._port_idx2name[np.argmin(vessel_plans[vessel_idx])], np.min(vessel_plans[vessel_idx]))

        # calculate vessel init time
        vessel_init_time = dict()
        for vessel_name, vessel_info in topo_config['vessels'].items():
            vessel_route_name = vessel_info['route']['route_name']
            vessel_speed = vessel_info['sailing']['speed']
            vessel_duration = vessel_info['parking']['duration']
            vessel_route = topo_config['routes'][vessel_route_name]

            vessel_init_time[vessel_name] = 0
            # calculate init time
            for port in vessel_route:                
                if port['port_name'] != vessel_next_stop[vessel_name][0]:
                    vessel_init_time[vessel_name] += round(port['distance'] / vessel_speed) + vessel_duration
                else:
                    vessel_init_time[vessel_name] += round(port['distance'] / vessel_speed) - vessel_next_stop[vessel_name][1] + current_tick
                    break

        # forecast vessel future arrvial
        for step in range(forecast_length):
            step_vessel_port_connection.setdefault(step, dict())
            for vessel_name, vessel_info in topo_config['vessels'].items():
                if step >= vessel_next_stop[vessel_name][1] - current_tick:
                    vessel_speed = vessel_info['sailing']['speed']
                    vessel_route_name = vessel_info['route']['route_name']
                    vessel_duration = vessel_info['parking']['duration']
                    vessel_route = topo_config['routes'][vessel_route_name]

                    relative_step = (step + vessel_init_time[vessel_name]) % self._vessel_cycle_time[vessel_name]

                    if relative_step in self._vessel_in_cycle_arrive_step[vessel_name].keys():
                        step_vessel_port_connection[step].setdefault(vessel_name, self._vessel_in_cycle_arrive_step[vessel_name][relative_step])

        # for step, v2p in step_vessel_port_connection.items():
        #     print(current_tick + step)
        #     for vessel, port in v2p.items():
        #         print(vessel, port)
        #     print()
        # exit()
        
        return step_vessel_port_connection

    def forecast_orders(self, forecast_length: int):    # moving average method need to be refined
        # make forecast_orders, forecast_port_orders as self.variables can avoid initialization multiple times
        forecast_orders = dict()
        forecast_port_orders = dict()

        for port_name in self._port_name_list:
            # Save history_order in dict: (key: port_name, value <list>: quantity needed each tick <0 if no order>)
            # then history_order_for_forcast = history_order[port_name][-moveing_average_length:]
            history_order_for_forecast = list()
            if port_name in self._order_history.keys():
                # exchange the order of 'for statement' & 'if statement'
                for tick in self._order_history[port_name].keys():
                    if tick > self._last_order_store_tick - self._moving_average_length:
                        history_order_for_forecast.append(self._order_history[port_name][tick])
                forecast_port_orders[port_name] = round(np.sum(history_order_for_forecast) / min(self._moving_average_length, self._last_order_store_tick) if self._last_order_store_tick > 0 else 1)
            else:
                forecast_port_orders[port_name] = 0

        # for port_name in self._port_name_list:
        #     if port_name in ['demand_port_001']:
        #         forecast_port_orders[port_name] = 500
        #     elif port_name in ['demand_port_002']:
        #         forecast_port_orders[port_name] = 500
        #     elif port_name in ['transfer_port_001']:
        #         forecast_port_orders[port_name] = 1000
        
        for step in range(forecast_length):
            forecast_orders.setdefault(step, dict())
            for port_name in self._port_name_list:
                forecast_orders[step][port_name] = forecast_port_orders[port_name]

        forecast_orders.setdefault(-1, dict())
        for port_name in self._port_name_list:
            forecast_orders[-1][port_name] = 0

        return forecast_orders

    # def forecast_return_empty(self, forecast_length: int):
    #     forecast_return_empty = dict()
    #     forecast_port_return_empty = dict()

    #     for port_name in self._port_name_list:
    #         return_empty_history_for_forecast = list()
    #         for tick in range(max(self._last_return_empty_store_tick - self._moving_average_length + 1, 0), self._last_return_empty_store_tick + 1):
    #             if tick in self._history_return_empty.keys():
    #                 return_empty_history_for_forecast.append(self._history_return_empty[port_name][tick])
    #             else:
    #                 return_empty_history_for_forecast.append(0)

    #         for step in range(forecast_length):
    #             forecast_return_empty.setdefault(step, dict())
    #             forecast_return_empty_one_step = round(np.average(return_empty_history_for_forecast[:])
    #             else:
    #                 forecast_return_empty[port_name] = 0
        
    #     for step in range(forecast_length):
    #         forecast_return_empty.setdefault(step, dict())
    #         for port_name in self._port_name_list:
    #             forecast_return_empty[step][port_name] = forecast_port_return_empty[port_name]

    #     forecast_return_empty.setdefault(-1, dict())
    #     for port_name in self._port_name_list:
    #         forecast_return_empty[-1][port_name] = 0
        
    #     return forecast_return_empty

    def forecast_return_empty(self, forecast_length: int):
        forecast_return_empty = dict()
        forecast_port_return_empty = dict()

        for port_name in self._port_name_list:
            history_return_empty_for_forecast = list()
            if port_name in self._return_empty_history.keys():
                for tick in self._return_empty_history[port_name].keys():
                    if tick > self._last_return_empty_store_tick - self._moving_average_length:
                        history_return_empty_for_forecast.append(self._return_empty_history[port_name][tick])
                forecast_port_return_empty[port_name] = round(np.sum(history_return_empty_for_forecast) / min(self._moving_average_length, self._last_return_empty_store_tick) if self._last_return_empty_store_tick > 0 else 1)
            else:
                forecast_port_return_empty[port_name] = 0
        
        for step in range(forecast_length):
            forecast_return_empty.setdefault(step, dict())
            for port_name in self._port_name_list:
                forecast_return_empty[step][port_name] = forecast_port_return_empty[port_name]

        forecast_return_empty.setdefault(-1, dict())
        for port_name in self._port_name_list:
            forecast_return_empty[-1][port_name] = 0
        
        return forecast_return_empty

    def forecast_vessel_full(self, forecast_length: int, vessel_full: dict, vessel_empty: dict, step_vessel_port_connection: dict):
        # Step 1: forecast vessel full delta
        forecast_full_delta = dict()

        for vessel_name in self._vessel_name_list:
            for port_name in self._port_name_list:
                history_full_delta_for_forecast = list()
                forecast_full_delta.setdefault(vessel_name, dict())

                if vessel_name in self._full_delta_history.keys() and port_name in self._full_delta_history[vessel_name].keys():
                    for tick in self._full_delta_history[vessel_name][port_name].keys():
                        if tick > self._last_full_delta_store_tick - self._moving_average_length:
                            history_full_delta_for_forecast.append(self._full_delta_history[vessel_name][port_name][tick])
                    forecast_full_delta[vessel_name][port_name] = round(np.average(history_full_delta_for_forecast)) if len(history_full_delta_for_forecast) > 0 else 0
                else:
                    forecast_full_delta[vessel_name][port_name] = 0

        # Step 2: calculate the forecasted vessel full with forecasted vessel full delta
        for step in range(forecast_length):
            for vessel_name in self._vessel_name_list:
                # TODO: no need for this if-statement
                if vessel_name in step_vessel_port_connection[step].keys():
                    step_vessel_full = vessel_full[step-1][vessel_name] + forecast_full_delta[vessel_name][step_vessel_port_connection[step][vessel_name]]
                else:
                    step_vessel_full = vessel_full[step-1][vessel_name]

                vessel_capacity = self._topo_config['vessels'][vessel_name]['capacity']
                vessel_full_upper_bound = vessel_capacity - vessel_empty[-1][vessel_name] if step == 0 else vessel_capacity

                step_vessel_full = max(step_vessel_full, 0)
                step_vessel_full = min(step_vessel_full, vessel_full_upper_bound)
                vessel_full[step][vessel_name] = step_vessel_full

        # print(self._history_full_delta)
        # for vessel_name in self._vessel_name_list:
        #     for port_name in self._port_name_list:
        #         # for tick in forecast_full_delta.keys():
        #         print(vessel_name, port_name, end='')
        #         print(' ', forecast_full_delta[vessel_name][port_name])

        # forecast_full_delta.setdefault(-1, dict())
        # for vessel_name in self._vessel_name_list:
        #     forecast_full_delta[-1][vessel_name] = 0
        # for vessel_name in self._vessel_name_list:
        #     print(vessel_name)
        #     for step in range(forecast_length):
        #         print('  ', step, vessel_full[step][vessel_name])

        # print actual and forecast line 

        return vessel_full
    
    def reset(self):
        self._order_history = dict() # [port][tick]
        self._return_empty_history = dict() # [port][tick]
        self._full_delta_history = dict()   # [vessel][tick]
        self._vessel_arrival_history = dict()

        self._last_order_store_tick = 0
        self._last_return_empty_store_tick = 0
        self._last_full_delta_store_tick = 0
        self._last_v2p_store_tick = 0
        
class Online_LP():
    def __init__(self, 
                 port_idx2name: list, 
                 vessel_idx2name: list, 
                 topo_config: list,
                 moving_average_length: int, 
                 window_size: int, 
                 apply_buffer_length: int,
                 time_decay: float,
                 order_gain_factor: float,
                 transit_cost_factor: float,
                 load_discharge_cost_factor: float
                 ):
        # LP related parameters
        self._window_size = window_size
        self._moving_average_length = moving_average_length
        self._time_decay = time_decay
        self._order_gain_factor = order_gain_factor
        self._transit_cost_factor = transit_cost_factor # delete 
        self._load_discharge_cost_factor = load_discharge_cost_factor

        # topo related variables
        self._port_idx2name = port_idx2name
        self._vessel_idx2name = vessel_idx2name
        self._topo_config = topo_config 
        self._port_name_list = list()
        self._vessel_name_list = list()
        for port_name in port_idx2name.values():
            self._port_name_list.append(port_name)
        for vessel_name in vessel_idx2name.values():
            self._vessel_name_list.append(vessel_name)
        
        # buffer related variables
        self._apply_buffer_length = apply_buffer_length
        self._vessel_applied_buffer_times = {vessel_name: 0 for vessel_name in self._vessel_name_list}
        
        #LP related constants
        self._port_capacity = dict()
        self._vessel_capacity = dict()
        for port_name, port_info in topo_config['ports'].items():
            self._port_capacity[port_name] = port_info['capacity']
        for vessel_name, vessel_info in topo_config['vessels'].items():
            self._vessel_capacity[vessel_name] = vessel_info['capacity']

        self._vessel_full = dict()

        # LP related variables
        self._port_empty = dict()
        self._vessel_empty = dict()

        # LP related decision variables
        self._discharge_empty = dict()
        self._load_empty = dict()
        self._order_apply = dict()

        # forecast module
        self._forecast = Forecast(self._moving_average_length, self._port_idx2name, self._vessel_idx2name, self._port_name_list, self._vessel_name_list, self._topo_config)

        self._initialize = True

    def _get_on_port_vessels(self, step: int, port_list: list):
        on_port_vessel_list = list()
        if step >= 0:
            for vessel_name, port_name in self._vessel_arrival[step].items():
                if port_name in port_list:
                    on_port_vessel_list.append(vessel_name)
        return on_port_vessel_list

    def _init_variables(self):
        for s in range(-1, self._window_size):
            self._port_empty[s] = dict()
            self._vessel_empty[s] = dict()
            self._vessel_full[s] = dict()
            self._order_apply[s] = dict()
            self._discharge_empty[s] = dict()
            self._load_empty[s] = dict()

            for p in self._port_name_list:
                self._port_empty[s][p] = LpVariable(name=f'port_empty__{s}_{p}', lowBound=0, cat='Integer')
                self._order_apply[s][p] = LpVariable(name=f'order_apply__{s}_{p}', lowBound=0, cat='Integer')
            
            on_port_vessels = self._get_on_port_vessels(s, self._port_name_list)
            for v in self._vessel_name_list:
                self._vessel_empty[s][v] = LpVariable(name=f'vessel_empty__{s}_{v}', lowBound=0, cat='Integer')
                if v in on_port_vessels:
                    self._load_empty[s][v] = LpVariable(name=f'load_empty__{s}_{v}', lowBound=0, cat='Integer')
                    self._discharge_empty[s][v] = LpVariable(name=f'discharge_empty__{s}_{v}', lowBound=0, cat='Integer')
                else:
                    self._load_empty[s][v] = 0
                    self._discharge_empty[s][v] = 0

    def _init_inventory(self, 
                        initial_port_empty: dict = None,
                        initial_vessel_empty: dict = None,
                        initial_vessel_full: dict = None,
                        ):
        for p in self._port_name_list:
            self._port_empty[-1][p] = initial_port_empty[p]
            self._order_apply[-1][p] = 0

        for v in self._vessel_name_list: 
            self._vessel_empty[-1][v] = initial_vessel_empty[v]
            self._vessel_full[-1][v] = np.sum([initial_vessel_full[v][p] for p in initial_vessel_full[v].keys()])

    def _add_constraints(self, problem): 
        for s in range(-1, self._window_size-1):
            # for port empty
            for p in self._port_name_list:
                on_port_vessel_list = self._get_on_port_vessels(s, [p])
                problem += self._port_empty[s+1][p] == self._port_empty[s][p] + self._return_empty[s][p] - self._order_apply[s][p] + lpSum([self._discharge_empty[s][v] - self._load_empty[s][v] for v in on_port_vessel_list]) 

            # for vessel empty
            for v in self._vessel_name_list:
                problem += self._vessel_empty[s+1][v] == self._vessel_empty[s][v] + self._load_empty[s][v] - self._discharge_empty[s][v]

        for s in range(0, self._window_size):
            # for capacity
            for p in self._port_name_list:
                problem += self._port_empty[s][p] <= self._port_capacity[p]
            for v in self._vessel_name_list:
                problem += self._vessel_empty[s][v] + self._vessel_full[s-1][v] <= self._vessel_capacity[v]

            # for apply order
            for p in self._port_name_list:
                problem += self._order_apply[s][p] <= self._orders[s][p]

    def _set_objective(self, problem):
        order_gain = self._order_gain_factor * lpSum([math.pow(self._time_decay, s) * self._order_apply[s][p] \
            for s in range(self._window_size) \
                for p in self._order_apply[s].keys()])

        transit_cost = self._transit_cost_factor * lpSum([self._vessel_empty[s][v] \
            for s in self._vessel_arrival.keys() \
                for v in self._vessel_name_list \
                    if self._vessel_arrival[s] == None and self._vessel_arrival[s][v] == None])
    
        load_discharge_cost = self._load_discharge_cost_factor * lpSum([self._load_empty[s][v] + self._discharge_empty[s][v] \
            for s in self._vessel_arrival.keys() \
                for v in self._vessel_arrival[s].keys()])
        
        problem += order_gain - transit_cost - load_discharge_cost
    
    def _forecast_data(self, finished_events, snapshot_list, current_tick):
        self._forecast.augment_history_record(current_tick, finished_events, snapshot_list)

        self._orders = self._forecast.forecast_orders(self._window_size)
        self._return_empty = self._forecast.forecast_return_empty(self._window_size)
        self._vessel_full = self._forecast.forecast_vessel_full(self._window_size, self._vessel_full, self._vessel_empty, self._vessel_arrival)
        # print(self._vessel_full)

    def _formulate_and_solve(self,
                             vessel_code: str,
                             finished_events: list, 
                             snapshot_list, 
                             current_tick: int,
                             initial_port_empty: dict,
                             initial_vessel_empty: dict,
                             initial_vessel_full: dict):
        self._vessel_arrival = self._forecast.calculate_vessel_arrival(self._topo_config, snapshot_list, current_tick, self._window_size)
        self._init_variables()
        self._init_inventory(initial_port_empty, initial_vessel_empty, initial_vessel_full)
        self._forecast_data(finished_events, snapshot_list, current_tick)
        
        problem = LpProblem(name=f"ecr Problem: from Tick_{current_tick}", sense=LpMaximize)

        self._add_constraints(problem)
        self._set_objective(problem)
     
        problem.solve()

        # print("STATUS", LpStatus[problem.status])
        # if LpStatus[problem.status] == 'Infeasible':
        #     for key, value in problem.constraints.items():
        #         if 'vessel_empty__3_rt1_vessel_002' in str(value) or 'vessel_empty__4_rt1_vessel_002' in str(value):
                # if value.slack != 0:
                    # print(key, value, value.slack)
        # print(self._vessel_full[0]['rt2_vessel_001'], self._vessel_full[1]['rt2_vessel_001'])

        #reset buffer
        self._vessel_applied_buffer_times = {vessel_name: 0 for vessel_name in self._vessel_name_list}
        self._decision_step_list = {vessel_name: list() for vessel_name in self._vessel_name_list}
        for s in self._vessel_arrival.keys():
            for vessel_name in self._vessel_arrival[s].keys():
                self._decision_step_list[vessel_name].append(s)
        for vessel_name in self._decision_step_list.keys():
            self._decision_step_list[vessel_name].sort()

    def choose_action(self,
                      current_tick: int,
                      port_code: str,
                      vessel_code: str,
                      finished_events: dict,
                      snapshot_list,
                      initial_port_empty: dict = None,
                      initial_vessel_empty: dict = None,
                      initial_vessel_full: dict = None,
                      ):
        if self._initialize:
            self._formulate_and_solve(vessel_code, finished_events, snapshot_list, current_tick, initial_port_empty, initial_vessel_empty, initial_vessel_full)
            self._initialize = False
        
        decision_step = self._find_next_decision_step(vessel_code)
        if decision_step >= self._apply_buffer_length:
            self._formulate_and_solve(vessel_code, finished_events, snapshot_list, current_tick, initial_port_empty, initial_vessel_empty, initial_vessel_full)
            decision_step = self._find_next_decision_step(vessel_code)

        return self._discharge_empty[decision_step][vessel_code].varValue - self._load_empty[decision_step][vessel_code].varValue

        #load full and discharge full sequence

    def _find_next_decision_step(self, vessel_code):
        if self._vessel_applied_buffer_times[vessel_code] < len(self._decision_step_list[vessel_code]):
            next_decision_step = self._decision_step_list[vessel_code][self._vessel_applied_buffer_times[vessel_code]]
        else:
            next_decision_step = self._apply_buffer_length
        self._vessel_applied_buffer_times[vessel_code] += 1
        return next_decision_step
        
    def reset(self):
        self._forecast.reset()