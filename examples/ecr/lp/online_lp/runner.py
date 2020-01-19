from collections import OrderedDict
from datetime import datetime

import io
import numpy as np
import os
import random
import torch
import yaml

from examples.ecr.lp.online_lp.online_lp_action_shaping import OnlineLPActionShaping
from examples.ecr.lp.online_lp.online_lp_agent import OnlineLPAgent
from examples.ecr.lp.online_lp.online_lp import Online_LP
from maro.simulator import Env
from maro.utils import Logger, LogFormat, convert_dottable

import maro.simulator.utils.random as srandom


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

MAX_EP = config.train.max_ep

SEED = config.train.seed
RUNNER_LOG_ENABLE = config.log.runner.enable

TIME_DECAY = config.online_lp.time_decay
WINDOW_SIZE = config.online_lp.window_size
APPLY_BUFFER_LENGTH = config.online_lp.apply_buffer_length
MOVING_AVERAGE_LENGTH = config.online_lp.moving_average_length
ORDER_GAIN_FACTOR = config.online_lp.order_gain_factor
TRANSIT_COST_FACTOR = config.online_lp.transit_cost_factor
LOAD_DISCHARGE_COST_FACTOR = config.online_lp.load_discharge_cost_factor

class Runner:
    def __init__(self, scenario: str, topology: str, max_tick: int, log_enable: bool = True):
        self._set_seed(SEED)
        self._env = Env(scenario, topology, max_tick)
        self._port_idx2name = self._env.node_name_mapping['static']
        self._vessel_idx2name = self._env.node_name_mapping['dynamic']
        self._online_lp_agent = self._load_agent()
        self._max_tick = max_tick
        self._log_enable = log_enable
        self._max_ep = MAX_EP

        if log_enable:
            self._logger = Logger(tag='runner', format_=LogFormat.simple,
                                  dump_folder=LOG_FOLDER, dump_mode='w', auto_timestamp=False)
            self._performance_logger = Logger(tag=f'runner.performance', format_=LogFormat.none,
                                              dump_folder=LOG_FOLDER, dump_mode='w', extension_name='csv',
                                              auto_timestamp=False)
            self._performance_logger.debug(
                f"episode,epsilon,{','.join([port_name + '_booking' for port_name in self._port_idx2name.values()])},total_booking,{','.join([port_name + '_shortage' for port_name in self._port_idx2name.values()])},total_shortage")
            
            self._port_logger = dict()
            for port in self._port_idx2name.keys():
                self._port_logger[port] = Logger(tag=f'{self._port_idx2name[port]}.logger', format_=LogFormat.none,
                                              dump_folder=LOG_FOLDER, dump_mode='w', extension_name='csv',
                                              auto_timestamp=False)
                self._port_logger[port].debug("tick, action, port_inventory, shortage")


    def _load_agent(self):
        action_shaping = OnlineLPActionShaping()
        online_lp = Online_LP(port_idx2name = self._port_idx2name, 
                              vessel_idx2name=self._vessel_idx2name, 
                              topo_config=self._env._business_engine.configs,
                              moving_average_length=MOVING_AVERAGE_LENGTH, 
                              window_size=WINDOW_SIZE, 
                              apply_buffer_length=APPLY_BUFFER_LENGTH,
                              time_decay=TIME_DECAY,
                              order_gain_factor=ORDER_GAIN_FACTOR,
                              transit_cost_factor=TRANSIT_COST_FACTOR,
                              load_discharge_cost_factor=LOAD_DISCHARGE_COST_FACTOR
                              )
        agent = OnlineLPAgent(online_lp, action_shaping, self._port_idx2name, self._vessel_idx2name)

        return agent

    def start(self):
        self._set_seed(SEED)

        _, decision_event, is_done = self._env.step(None)

        while not is_done:
            initial_port_empty, initial_vessel_empty, initial_vessel_full = self._get_initial_values()

            action = self._online_lp_agent.choose_action(decision_event=decision_event,
                                                        finished_events=self._env._event_buffer.get_finished_events(),
                                                        snapshot_list=self._env.snapshot_list,
                                                        initial_port_empty=initial_port_empty,
                                                        initial_vessel_empty=initial_vessel_empty,
                                                        initial_vessel_full=initial_vessel_full
                                                        )
            
            ports = self._env.snapshot_list.static_nodes
            self._port_logger[decision_event.port_idx].info(f"{decision_event.tick}, {action.quantity}, {ports[decision_event.tick:decision_event.port_idx: ('empty', 0)][0]}, {np.sum(ports[: decision_event.port_idx: ('shortage', 0)])}")

            # action = Action(decision_event.vessel_idx, decision_event.port_idx, random.randint(-decision_event.action_scope.load, decision_event.action_scope.discharge))
            _, decision_event, is_done = self._env.step(action)

        self._print_summary()

        self._env.reset()

        self._online_lp_agent.reset()

    def _get_initial_values(self):
        initial_port_empty = {
            port.name: port.empty
            for port in self._env._business_engine._ports
        }
        
        initial_vessel_empty = {
            vessel.name: vessel.empty
            for vessel in self._env._business_engine._vessels
        }

        initial_vessel_full = {
            vessel.name: {
                port.name: self._env._business_engine._full_on_vessels[vessel.idx: port.idx] 
                for port in self._env._business_engine._ports
            }
            for vessel in self._env._business_engine._vessels
        }
        
        return initial_port_empty, initial_vessel_empty, initial_vessel_full

    def _print_summary(self):
        shortage_list = self._env.snapshot_list.static_nodes[
                        self._env.tick: self._env.agent_idx_list: ('acc_shortage', 0)]
        pretty_shortage_dict = OrderedDict()
        tot_shortage = 0
        for i, shortage in enumerate(shortage_list):
            pretty_shortage_dict[self._port_idx2name[i]] = shortage
            tot_shortage += shortage
        pretty_shortage_dict['total_shortage'] = tot_shortage

        booking_list = self._env.snapshot_list.static_nodes[
                       self._env.tick: self._env.agent_idx_list: ('acc_booking', 0)]
        pretty_booking_dict = OrderedDict()
        tot_booking = 0
        for i, booking in enumerate(booking_list):
            pretty_booking_dict[self._port_idx2name[i]] = booking
            tot_booking += booking
        pretty_booking_dict['total_booking'] = tot_booking

        self._logger.critical(
            f'{self._env.name} | test | total tick: {self._max_tick}, total booking: {pretty_booking_dict}, total shortage: {pretty_shortage_dict}')
        
        last_224_shortage = np.sum(shortage_list) - np.sum(self._env.snapshot_list.static_nodes[self._env.tick - 224: self._env.agent_idx_list: ('acc_shortage', 0)])
        print(f'last 224 shortage: {last_224_shortage}')


    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        srandom.seed(seed)


if __name__ == '__main__':
    runner = Runner(scenario=SCENARIO, topology=TOPOLOGY, max_tick=MAX_TICK, log_enable=RUNNER_LOG_ENABLE)

    runner.start()
