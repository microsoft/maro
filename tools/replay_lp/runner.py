# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from datetime import datetime
import io
import os
import random

import numpy as np
import torch
import yaml

import maro.simulator.utils.random as srandom

from maro.simulator import Env
from maro.utils import Logger, LogFormat

from maro.simulator.scenarios.ecr.common import Action, EcrEventType
from maro.utils import convert_dottable

from tools.replay_lp.agent import LPAgent
from tools.replay_lp.lp import LPReplayer
from tools.replay_lp.action_shaping import LPActionShaping

CONFIG_PATH = os.environ.get('CONFIG') or 'config.yml'

with io.open(CONFIG_PATH, 'r') as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

LOG_FOLDER = os.path.join(os.getcwd(), 'log', f"{datetime.now().strftime('%Y%m%d')}", config.experiment_name)
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

with io.open(os.path.join(LOG_FOLDER, 'config.yml'), 'w', encoding='utf8') as out_file:
    yaml.safe_dump(raw_config, out_file)

SCENARIO = config.env.scenario
TOPOLOGY = config.env.topology
MAX_TICK = config.env.max_tick

SEED = config.seed
LP_CONFIG = config.lp

class Runner:
    def __init__(self, scenario: str, topology: str, max_tick: int):
        self._scenario = scenario
        self._topology = topology
        self._max_tick = max_tick

        self._init_env(self._max_tick)

        self._logger = Logger(tag='runner', format_=LogFormat.none, dump_folder=LOG_FOLDER,
                              dump_mode='w', auto_timestamp=False, extension_name='txt')
        
        self._lp_action_logger = Logger(tag='lp_env_action', format_=LogFormat.none, dump_folder=LOG_FOLDER,
                                        dump_mode='w', auto_timestamp=False, extension_name='csv')
        self._lp_action_logger.debug(f'tick,port,vessel,action_scope.discharge,action_scope.load,lp_env_action')

    def _init_env(self, max_tick: int):
        self._set_seed(SEED)
        self._env = Env(self._scenario, self._topology, max_tick)

        self._business_engine = self._env._business_engine
        self._event_buffer = self._env._event_buffer

        self._port_idx2name = self._env.node_name_mapping['static']
        self._vessel_idx2name = self._env.node_name_mapping['dynamic']

    def start(self):
        self._set_seed(SEED)
        self._dummy_interaction()
        orders, tick_vessel_port_connection = self._record()
        self._env.reset()

        self._set_seed(SEED)
        self._lp = self._load_lp_agent(orders, tick_vessel_port_connection)
        self._set_seed(SEED)
        self._replay()
        orders2, tick_vessel_port_connection2 = self._record()

        self._env_info_compare(orders, orders2, tick_vessel_port_connection, tick_vessel_port_connection2)

    def _env_info_compare(self, order1, order2, connection1, connection2):
        assert len(order1.keys()) == len(order2.keys()), f'total order tick num differ!'
        for tick in order1.keys():
            assert tick in order2, f'lost tick {tick} in order2!'
            assert len(order1[tick].keys()) == len(order2[tick].keys()), f'total order src num of tick {tick} differ!'
            for src in order1[tick].keys():
                assert src in order2[tick], f'lost src {src} in order2[{tick}]!'
                assert len(order1[tick][src].keys()) == len(order2[tick][src].keys()), f'total order dst num of tick {tick} src {src} differ!'
                for dest in order1[tick][src].keys():
                    assert dest in order2[tick][src], f'lost dest {dest} in order2[{tick}][{src}]!'
                    assert order1[tick][src][dest] == order2[tick][src][dest]

        assert len(connection1.keys()) == len(connection2.keys()), f'total connection tick num differ!'
        for tick in connection1.keys():
            assert tick in connection2, f'lost tick {tick} in tick_vessel_port_connection2!'
            assert len(connection1[tick].keys()) == len(connection2[tick].keys()), f'total connection vessel num differ!'
            for vessel in connection1[tick].keys():
                assert vessel in connection2[tick], f'lost vessel {vessel} in tick_vessel_port_connection2[{tick}]!'
                assert connection1[tick][vessel] == connection2[tick][vessel]

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        srandom.seed(seed)

    def _dummy_interaction(self):
        """
            Interact with env using random action
        """
        _, decision_event, is_done = self._env.step(None)
        while not is_done:
            random_action = random.randint(-decision_event.action_scope.load, decision_event.action_scope.discharge)
            env_action = Action(decision_event.vessel_idx, decision_event.port_idx, random_action)
            _, decision_event, is_done = self._env.step(env_action)

        self._print_summary(summary_name='Random action test')

    def _record(self):
        """
            Process finished event buffer, get orders and vessel arrival info from it
        """
        finished_events = self._event_buffer.get_finished_events()
        orders = dict()
        tick_vessel_port_connection = dict()
        for event in finished_events:
            if event.event_type == EcrEventType.ORDER:
                tick = event.payload.tick
                src = self._port_idx2name[event.payload.src_port_idx]
                dest = self._port_idx2name[event.payload.dest_port_idx]
                qty = event.payload.quantity
                orders.setdefault(tick, dict())
                orders[tick].setdefault(src, dict())
                orders[tick][src].setdefault(dest, 0)
                orders[tick][src][dest] += qty
            elif event.event_type == EcrEventType.LOAD_FULL:
                tick = event.tick
                port_name = self._port_idx2name[event.payload.port_idx]
                vessel_name = self._vessel_idx2name[event.payload.vessel_idx]
                tick_vessel_port_connection.setdefault(tick, dict())
                tick_vessel_port_connection[tick][vessel_name] = port_name
        return orders, tick_vessel_port_connection

    def _load_lp_agent(self, orders, tick_vessel_port_connection):
        """
            1. Read static info from BE and initialize LP Agent
            2. Read status info from BE and formulate LP problem
        """
        port_list, vessel_list, port_capacity, vessel_capacity, vessel_routes, full_return_buffer_ticks, empty_return_buffer_ticks = self._get_static_info()        

        action_shaping = LPActionShaping()
        lp = LPReplayer(configs=LP_CONFIG,
                        log_folder=LOG_FOLDER,
                        port_list=port_list,
                        vessel_list=vessel_list,
                        port_capacity=port_capacity,
                        vessel_capacity=vessel_capacity,
                        vessel_routes=vessel_routes,
                        full_return_buffer_ticks=full_return_buffer_ticks,
                        empty_return_buffer_ticks=empty_return_buffer_ticks,
                        orders=orders,
                        tick_vessel_port_connection=tick_vessel_port_connection,
                        )
        agent = LPAgent(lp, action_shaping, self._port_idx2name, self._vessel_idx2name)

        # Only formulate at the beginning is right, for the initial values got are not same as the beginning of this tick
        initial_port_empty, initial_port_on_consignee, initial_port_full, initial_vessel_empty, initial_vessel_full = self._get_initial_values()
        lp.formulate_and_solve(current_tick=0,
                               initial_port_empty=initial_port_empty,
                               initial_port_on_consignee=initial_port_on_consignee,
                               initial_port_full=initial_port_full,
                               initial_vessel_empty=initial_vessel_empty,
                               initial_vessel_full=initial_vessel_full
                               )
        return agent
    
    def _get_static_info(self):
        """
            Read static info from BE, and return them for LP agent's initialization
        """
        configs = self._business_engine.configs

        # Constant value
        port_list = [port.name for port in self._business_engine._ports]
        vessel_list = [vessel.name for vessel in self._business_engine._vessels]
        port_capacity = dict()
        vessel_capacity = dict()
        vessel_routes = dict()

        # Expected value of random variables
        full_return_buffer_ticks = dict()
        empty_return_buffer_ticks = dict()

        for port in self._business_engine._ports:
            port_capacity[port.name] = port.capacity
            full_return_buffer_ticks[port.name] = configs['ports'][port.name]['full_return']['buffer_ticks']
            empty_return_buffer_ticks[port.name] = configs['ports'][port.name]['empty_return']['buffer_ticks']

        for vessel in self._business_engine._vessels:
            vessel_capacity[vessel.name] = vessel.capacity

        routes = dict()
        for name, stop_list in configs['routes'].items():
            routes[name] = [stop['port_name'] for stop in stop_list]
        for vessel_name, vessel_info in configs['vessels'].items():
            vessel_routes[vessel_name] = routes[vessel_info['route']['route_name']]
        
        return port_list, vessel_list, port_capacity, vessel_capacity, vessel_routes, full_return_buffer_ticks, empty_return_buffer_ticks

    def _get_initial_values(self):
        """
            Read status value from BE, and return them for LP problem formulation
        """
        initial_port_empty = {
            port.name: port.empty
            for port in self._business_engine._ports
        }

        initial_port_on_consignee = {
            port.name: port.on_consignee
            for port in self._business_engine._ports
        }

        initial_port_full = {
            port.name: {
                dest.name: self._business_engine._full_on_ports[port.idx: dest.idx]
                for dest in self._business_engine._ports
            }
            for port in self._business_engine._ports
        }

        initial_vessel_empty = {
            vessel.name: vessel.empty
            for vessel in self._business_engine._vessels
        }

        initial_vessel_full = {
            vessel.name: {
                port.name: self._business_engine._full_on_vessels[vessel.idx: port.idx] 
                for port in self._business_engine._ports
            }
            for vessel in self._business_engine._vessels
        }

        return initial_port_empty, initial_port_on_consignee, initial_port_full, initial_vessel_empty, initial_vessel_full

    def _replay(self):
        """
            Replay with environment, directly use the action got from LP's formulation
        """
        _, decision_event, is_done = self._env.step(None)

        while not is_done:
            action = self._lp.choose_action(decision_event=decision_event)
            self._lp_action_logger.debug(f'{decision_event.tick},{self._port_idx2name[decision_event.port_idx]},{self._vessel_idx2name[decision_event.vessel_idx]},{decision_event.action_scope.discharge},{decision_event.action_scope.load},{action.quantity}')
            _, decision_event, is_done = self._env.step(action)

        # Print summary
        self._print_summary(summary_name='LP Replay')

    def _print_summary(self, summary_name=None):
        """
            Print the summary of shortage info of this episode
        """
        shortage_list = self._env.snapshot_list.static_nodes[
                        self._env.tick: self._env.agent_idx_list: ('acc_shortage', 0)]
        booking_list = self._env.snapshot_list.static_nodes[
                       self._env.tick: self._env.agent_idx_list: ('acc_booking', 0)]

        tot_shortage, tot_booking = 0, 0
        self._logger.info(f'******************** {summary_name} ********************')
        self._logger.info(f'[Port]: Shortage / Booking')
        for i in range(len(shortage_list)):
            name = self._port_idx2name[i]
            self._logger.info(f'[{name}]: {shortage_list[i]} / {booking_list[i]}')
            tot_shortage += shortage_list[i]
            tot_booking += booking_list[i]
        self._logger.info(f'[Total]: {tot_shortage} / {tot_booking}')


if __name__ == '__main__':
    runner = Runner(scenario=SCENARIO, topology=TOPOLOGY, max_tick=MAX_TICK)
    runner.start()