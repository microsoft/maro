import numpy as np

from maro.simulator.scenarios.cim.events import Events


INVALID_TICK = -1


class Forecaster:
    def __init__(
        self,
        moving_average_length: int,
        port_idx2name: dict,
        vessel_idx2name: dict,
        topology_config: dict
    ):
        self._moving_average_length = moving_average_length
        self._port_idx2name = port_idx2name
        self._vessel_idx2name = vessel_idx2name
        self._topology_config = topology_config

        self._port_idx_list: list = list(self._port_idx2name.keys())
        self._port_name_list = list(self._port_idx2name.values())
        self._vessel_idx_list = list(self._vessel_idx2name.keys())
        self._vessel_name_list = list(self._vessel_idx2name.values())

        self._num_ports = len(self._port_idx_list)
        self._num_vessels = len(self._vessel_idx_list)

        self._vessel_cycle_time, self._vessel_in_cycle_arrive_step = self._calculate_vessel_related_time()

        self._order_history = dict()            # [port][tick]
        self._return_empty_history = dict()     # [port][tick]
        self._full_delta_history = dict()       # [vessel][port][tick]
        self._vessel_arrival_history = dict()   # [tick][vessel][port]

        self._last_order_store_tick = 0
        self._last_return_empty_store_tick = 0
        self._last_full_delta_store_tick = 0
        self._last_v2p_store_tick = 0

    def augment_history_record(self, current_tick, finished_events, snapshot_list):
        last_order_tick = INVALID_TICK
        last_return_empty_tick = INVALID_TICK
        last_v2p_tick = INVALID_TICK

        for event in finished_events:
            if event.event_type == Events.ORDER:
                if event.payload.tick > self._last_order_store_tick:
                    if event.payload.tick > last_order_tick:
                        last_order_tick = event.payload.tick
                    tick = event.payload.tick
                    port = self._port_idx2name[event.payload.src_port_idx]
                    qty = event.payload.quantity
                    self._order_history.setdefault(port, dict())
                    self._order_history[port].setdefault(tick, 0)
                    self._order_history[port][tick] += qty

            elif event.event_type == Events.RETURN_EMPTY:
                if event.tick > self._last_return_empty_store_tick:
                    if event.tick > last_return_empty_tick:
                        last_return_empty_tick = event.tick
                    tick = event.tick
                    port = self._port_idx2name[event.payload.port_idx]
                    qty = event.payload.quantity
                    self._return_empty_history.setdefault(port, dict())
                    self._return_empty_history[port].setdefault(tick, 0)
                    self._return_empty_history[port][tick] += qty

            elif event.event_type == Events.LOAD_FULL:
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

        # need to have an event on load full
        pre_full_on_vessels = []
        for tick in range(self._last_full_delta_store_tick, current_tick):
            full_on_vessels = snapshot_list["matrices"][tick::"full_on_vessels"].reshape(
                self._num_vessels, self._num_ports
            )
            if pre_full_on_vessels != []:
                for vessel_idx in self._vessel_idx_list:
                    vessel_name = self._vessel_idx2name[vessel_idx]
                    if (
                        tick in self._vessel_arrival_history.keys()
                        and vessel_name in self._vessel_arrival_history[tick].keys()
                    ):
                        port_name = self._vessel_arrival_history[tick][vessel_name]
                        self._full_delta_history.setdefault(vessel_name, dict())
                        self._full_delta_history[vessel_name].setdefault(port_name, dict())
                        self._full_delta_history[vessel_name][port_name][tick] = (
                            np.sum(full_on_vessels[vessel_idx]) - np.sum(pre_full_on_vessels[vessel_idx])
                        )
            pre_full_on_vessels = full_on_vessels
        self._last_full_delta_store_tick = current_tick - 1

    def _calculate_vessel_related_time(self):
        vessel_in_cycle_arrive_step = dict()
        vessel_cycle_time = dict()

        for vessel_name, vessel_info in self._topology_config["vessels"].items():
            vessel_route_name = vessel_info["route"]["route_name"]
            sailing_speed = vessel_info["sailing"]["speed"]
            parking_duration = vessel_info["parking"]["duration"]
            vessel_route = self._topology_config["routes"][vessel_route_name]

            vessel_in_cycle_arrive_step.setdefault(vessel_name, dict())
            vessel_cycle_time[vessel_name] = 0
            # calculate cycle time
            for port in vessel_route:
                vessel_cycle_time[vessel_name] += (
                    round(port["distance_to_next_port"] / sailing_speed) + parking_duration
                )

                vessel_in_cycle_arrive_step[vessel_name][
                    vessel_cycle_time[vessel_name] - parking_duration
                ] = port["port_name"]

        return vessel_cycle_time, vessel_in_cycle_arrive_step

    def calculate_vessel_arrival(self, snapshot_list, current_tick, forecast_length):
        step_vessel_port_connection = dict()
        vessel_plans = snapshot_list["matrices"][current_tick::"vessel_plans"].reshape(
            self._num_vessels, self._num_ports
        )

        # calculate vessel next stop port
        vessel_next_stop = dict()
        for vessel_idx in self._vessel_idx_list:
            for port_idx in self._port_idx_list:
                if vessel_plans[vessel_idx][port_idx] == -1:
                    vessel_plans[vessel_idx][port_idx] = 1000000  # TODO: change to max int
            vessel_next_stop[self._vessel_idx2name[vessel_idx]] = (
                self._port_idx2name[np.argmin(vessel_plans[vessel_idx])],
                np.min(vessel_plans[vessel_idx])
            )

        # calculate vessel init time
        vessel_init_time = dict()
        for vessel_name, vessel_info in self._topology_config["vessels"].items():
            vessel_route_name = vessel_info["route"]["route_name"]
            sailing_speed = vessel_info["sailing"]["speed"]
            parking_duration = vessel_info["parking"]["duration"]
            vessel_route = self._topology_config["routes"][vessel_route_name]

            vessel_init_time[vessel_name] = 0
            # calculate init time
            for port in vessel_route:
                if port["port_name"] != vessel_next_stop[vessel_name][0]:
                    vessel_init_time[vessel_name] += (
                        round(port["distance_to_next_port"] / sailing_speed) + parking_duration
                    )
                else:
                    vessel_init_time[vessel_name] += (
                        round(port["distance_to_next_port"] / sailing_speed)
                        - vessel_next_stop[vessel_name][1]
                        + current_tick
                    )
                    break

        # forecast vessel future arrvial
        for step in range(forecast_length):
            step_vessel_port_connection.setdefault(step, dict())
            for vessel_name, vessel_info in self._topology_config["vessels"].items():
                if step >= vessel_next_stop[vessel_name][1] - current_tick:
                    relative_step = (step + vessel_init_time[vessel_name]) % self._vessel_cycle_time[vessel_name]
                    if relative_step in self._vessel_in_cycle_arrive_step[vessel_name].keys():
                        step_vessel_port_connection[step].setdefault(
                            vessel_name,
                            self._vessel_in_cycle_arrive_step[vessel_name][relative_step]
                        )

        return step_vessel_port_connection

    def forecast_orders(self, forecast_length: int):
        # moving average method need to be refined
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
                forecast_port_orders[port_name] = round(
                    np.sum(history_order_for_forecast) / min(self._moving_average_length, self._last_order_store_tick)
                    if self._last_order_store_tick > 0
                    else 1
                )
            else:
                forecast_port_orders[port_name] = 0

        for step in range(forecast_length):
            forecast_orders.setdefault(step, dict())
            for port_name in self._port_name_list:
                forecast_orders[step][port_name] = forecast_port_orders[port_name] * 1.1

        forecast_orders.setdefault(-1, dict())
        for port_name in self._port_name_list:
            forecast_orders[-1][port_name] = 0

        return forecast_orders

    def forecast_return_empty(self, forecast_length: int):
        forecast_return_empty = dict()
        forecast_port_return_empty = dict()

        for port_name in self._port_name_list:
            history_return_empty_for_forecast = list()
            if port_name in self._return_empty_history.keys():
                for tick in self._return_empty_history[port_name].keys():
                    if tick > self._last_return_empty_store_tick - self._moving_average_length:
                        history_return_empty_for_forecast.append(self._return_empty_history[port_name][tick])
                forecast_port_return_empty[port_name] = round(
                    np.sum(history_return_empty_for_forecast) / min(
                        self._moving_average_length,
                        self._last_return_empty_store_tick
                    )
                    if self._last_return_empty_store_tick > 0
                    else 1
                )
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

    def forecast_vessel_full_delta(self, forecast_length: int):
        forecast_full_delta = dict()

        for vessel_name in self._vessel_name_list:
            for port_name in self._port_name_list:
                history_full_delta_for_forecast = list()
                forecast_full_delta.setdefault(vessel_name, dict())

                if (
                    vessel_name in self._full_delta_history.keys()
                    and port_name in self._full_delta_history[vessel_name].keys()
                ):
                    for tick in self._full_delta_history[vessel_name][port_name].keys():
                        if tick > self._last_full_delta_store_tick - self._moving_average_length:
                            history_full_delta_for_forecast.append(
                                self._full_delta_history[vessel_name][port_name][tick]
                            )
                    forecast_full_delta[vessel_name][port_name] = (
                        round(np.average(history_full_delta_for_forecast))
                        if len(history_full_delta_for_forecast) > 0
                        else 0
                    )
                else:
                    forecast_full_delta[vessel_name][port_name] = 0

        return forecast_full_delta

    def reset(self):
        self._order_history = dict()            # [port][tick]
        self._return_empty_history = dict()     # [port][tick]
        self._full_delta_history = dict()       # [vessel][tick]
        self._vessel_arrival_history = dict()

        self._last_order_store_tick = 0
        self._last_return_empty_store_tick = 0
        self._last_full_delta_store_tick = 0
        self._last_v2p_store_tick = 0
