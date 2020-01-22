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

from examples.ecr.lp.offline_lp.offline_lp_agent import OfflineLPAgent as LPAgent
from examples.ecr.lp.offline_lp.offline_lp import LPReplayer
from examples.ecr.lp.offline_lp.offline_lp_action_shaping import OfflineLPActionShaping as LPActionShaping

from examples.ecr.lp.online_lp.online_lp_agent import OnlineLPAgent
from examples.ecr.lp.online_lp.online_lp import Online_LP
from examples.ecr.lp.online_lp.online_lp_action_shaping import OnlineLPActionShaping

from examples.ecr.q_learning.common.agent import Agent
from examples.ecr.q_learning.common.dqn import QNet, DQN
from examples.ecr.common.state_shaping import StateShaping
from examples.ecr.common.action_shaping import DiscreteActionShaping

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

LP_CONFIG = config.lp
TRAIN_SEED = config.train.seed
TEST_SEED = config.test.seed

ENABLE_DQN = config.comparison.dqn.enable
DQN_MODEL_FOLDER = config.comparison.dqn.model_folder
DQN_MODEL_PATH_SUFFIX = config.comparison.dqn.model_path_suffix

TIME_DECAY = config.online_lp.time_decay
WINDOW_SIZE = config.online_lp.window_size
APPLY_BUFFER_LENGTH = config.online_lp.apply_buffer_length
MOVING_AVERAGE_LENGTH = config.online_lp.moving_average_length
ORDER_GAIN_FACTOR = config.online_lp.order_gain_factor
TRANSIT_COST_FACTOR = config.online_lp.transit_cost_factor
LOAD_DISCHARGE_COST_FACTOR = config.online_lp.load_discharge_cost_factor

class Runner:
    def __init__(self,
                 scenario: str,
                 topology: str,
                 max_tick: int,
                 ):
        self._set_seed(TRAIN_SEED)
        self._env = Env(scenario, topology, max_tick)
        self._business_engine = self._env._business_engine
        self._event_buffer = self._env._event_buffer

        self._port_idx2name = self._env.node_name_mapping['static']
        self._vessel_idx2name = self._env.node_name_mapping['dynamic']

        self._logger = Logger(tag='runner', format_=LogFormat.none, dump_folder=LOG_FOLDER,
                              dump_mode='w', auto_timestamp=False, extension_name='txt')
        
        self._lp_action_logger = Logger(tag='lp_env_action', format_=LogFormat.none, dump_folder=LOG_FOLDER,
                                        dump_mode='w', auto_timestamp=False, extension_name='csv')
        self._lp_action_logger.debug(f'tick,port,vessel,action_scope.discharge,action_scope.load,lp_env_action')

    def start(self):
        self._set_seed(TRAIN_SEED)
        self._dummy_interaction()
        orders, tick_vessel_port_connection = self._record()
        self._env.reset()

        self._online_lp = self._load_online_lp_agent()
        self._set_seed(TRAIN_SEED)
        self._online_lp_iteraction()
        orders2, tick_vessel_port_connection2 = self._record()
        self._env.reset()

        self._lp = self._load_agent(orders, tick_vessel_port_connection)
        self._set_seed(TRAIN_SEED)
        self._replay()
        orders3, tick_vessel_port_connection3 = self._record()
        self._env.reset()

        if ENABLE_DQN:
            self._dqn_agents = self._load_dqn_agents(self._env.agent_idx_list)
            self._set_seed(TRAIN_SEED)
            self._dqn_test()

        # self._compare("random v.s. online lp", orders, orders2, tick_vessel_port_connection, tick_vessel_port_connection2)
        # self._compare("random v.s. offline lp", orders, orders3, tick_vessel_port_connection, tick_vessel_port_connection3)

    def _compare(self, comparison_name, orders, orders2, tick_vessel_port_connection, tick_vessel_port_connection2):
        for tick in orders.keys():
            assert (tick in orders2), f'{comparison_name}: lost tick {tick} in orders2!'
            for src in orders[tick].keys():
                assert (src in orders2[tick]), f'{comparison_name}: lost src {src} in orders2[{tick}]!'
                for dest in orders[tick][src].keys():
                    assert (dest in orders2[tick][src]), f'{comparison_name}: lost dest {dest} in orders2[{tick}][{src}]!'
                    assert (orders[tick][src][dest] == orders2[tick][src][dest])
        for tick in tick_vessel_port_connection.keys():
            assert (tick in tick_vessel_port_connection2), f'{comparison_name}: lost tick {tick} in tick_vessel_port_connection2!'
            for vessel in tick_vessel_port_connection[tick].keys():
                assert (vessel in tick_vessel_port_connection2[tick]), f'{comparison_name}: lost vessel {vessel} in tick_vessel_port_connection2[{tick}]!'
                assert (tick_vessel_port_connection[tick][vessel] == tick_vessel_port_connection2[tick][vessel])

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        srandom.seed(seed)

    def _dummy_interaction(self):
        _, decision_event, is_done = self._env.step(None)

        while not is_done:
            random_action = random.randint(-decision_event.action_scope.load, decision_event.action_scope.discharge)
            env_action = Action(decision_event.vessel_idx, decision_event.port_idx, random_action)

            srandom.seed(decision_event.tick)

            _, decision_event, is_done = self._env.step(env_action)

        self._print_summary(summary_name='Random action test')

    def _record(self):
        # Process finished event buffer and get orders & vessel_arrival
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

    def _load_agent(self, orders, tick_vessel_port_connection):
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
        initial_port_empty, initial_port_on_shipper, initial_port_on_consignee, initial_port_full, initial_vessel_empty, initial_vessel_full = self._get_initial_values()
        lp.formulate_and_solve(current_tick=0,
                               initial_port_empty=initial_port_empty,
                               initial_port_on_shipper=initial_port_on_shipper,
                               initial_port_on_consignee=initial_port_on_consignee,
                               initial_port_full=initial_port_full,
                               initial_vessel_empty=initial_vessel_empty,
                               initial_vessel_full=initial_vessel_full
                               )
        return agent

    def _load_online_lp_agent(self):
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

    def _online_lp_iteraction(self):
        _, decision_event, is_done = self._env.step(None)

        while not is_done:
            initial_port_empty, _, _, _, initial_vessel_empty, initial_vessel_full = self._get_initial_values()

            action = self._online_lp.choose_action(decision_event=decision_event,
                                                         finished_events=self._env._event_buffer.get_finished_events(),
                                                         snapshot_list=self._env.snapshot_list,
                                                         initial_port_empty=initial_port_empty,
                                                         initial_vessel_empty=initial_vessel_empty,
                                                         initial_vessel_full=initial_vessel_full
                                                         )

            srandom.seed(decision_event.tick)

            _, decision_event, is_done = self._env.step(action)
        
        self._print_summary(summary_name='Online LP test')

    def _load_dqn_agents(self, agent_idx_list: [int]):
        agent_dict = {}
        state_shaping = StateShaping(env=self._env,
                                     relative_tick_list=[-1, -2, -3, -4, -5, -6, -7],
                                     port_downstream_max_number=2,
                                     port_attribute_list=['empty', 'full', 'on_shipper', 'on_consignee', 'booking', 'shortage', 'fulfillment'],
                                     vessel_attribute_list=['empty', 'full', 'remaining_space'])
        action_space = [round(i * 0.1, 1) for i in range(-10, 11)]
        action_shaping = DiscreteActionShaping(action_space=action_space)
        for agent_idx in agent_idx_list:
            policy_net = QNet(name=f'{self._port_idx2name[agent_idx]}.policy', input_dim=state_shaping.dim, hidden_dims=[256, 128, 64], output_dim=len(action_space),
                              dropout_p=0.0, log_enable=False, log_folder=LOG_FOLDER)
            dqn = DQN(policy_net=policy_net, target_net=policy_net,
                      gamma=0.0, tau=0.1, target_update_frequency=5, lr=0.05, log_enable=False, log_folder=LOG_FOLDER, log_dropout_p=0.95)
            agent_dict[agent_idx] = Agent(agent_name=self._port_idx2name[agent_idx], topology=TOPOLOGY,
                                          port_idx2name=self._port_idx2name, vessel_idx2name=self._vessel_idx2name,
                                          algorithm=dqn, experience_pool=None, state_shaping=state_shaping, action_shaping=action_shaping, reward_shaping=None,
                                          batch_num=0, batch_size=0, min_train_experience_num=0, log_enable=False, log_folder=LOG_FOLDER)
            model_path = os.path.join(DQN_MODEL_FOLDER, f'{self._port_idx2name[agent_idx]}{DQN_MODEL_PATH_SUFFIX}')
            agent_dict[agent_idx].load_policy_net_parameters(torch.load(model_path))

        return agent_dict
    
    def _dqn_test(self):
        _, decision_event, is_done = self._env.step(None)

        action_space = [round(i * 0.1, 1) for i in range(-10, 11)]
        action_shaping = DiscreteActionShaping(action_space=action_space)
        while not is_done:
            action = self._dqn_agents[decision_event.port_idx].choose_action(decision_event=decision_event, eps=0, current_ep=0)
            _, decision_event, is_done = self._env.step(action)
        
        self._print_summary(summary_name='DQN test')
    
    def _get_initial_values(self, current_tick: int = 0):
        initial_port_empty = {
            port.name: port.empty
            for port in self._business_engine._ports
        }

        initial_port_on_shipper = {
            p1.name: {
                p2.name: 0 for p2 in self._business_engine._ports
            }
            for p1 in self._business_engine._ports
        }
        # TODO: hard-coded buffer tick here
        pending_events = self._business_engine._event_buffer.get_pending_events(current_tick + 1)
        for pending_event in pending_events:
            if pending_event == None:
                continue
            if pending_event.event_type == EcrEventType.RETURN_FULL:
                src_port_idx, dst_port_idx, qty = pending_event.payload
                src_port_name = self._port_idx2name[src_port_idx]
                dst_port_name = self._port_idx2name[dst_port_idx]
                initial_port_on_shipper[src_port_name][dst_port_name] = qty

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
        
        return initial_port_empty, initial_port_on_shipper, initial_port_on_consignee, initial_port_full, initial_vessel_empty, initial_vessel_full

    def _replay(self):
        _, decision_event, is_done = self._env.step(None)

        while not is_done:
            initial_port_empty, initial_port_on_shipper, initial_port_on_consignee, initial_port_full, initial_vessel_empty, initial_vessel_full = self._get_initial_values(decision_event.tick)


            action = self._lp.choose_action(decision_event=decision_event,
                                            initial_port_empty=initial_port_empty,
                                            initial_port_on_shipper=initial_port_on_shipper,
                                            initial_port_on_consignee=initial_port_on_consignee,
                                            initial_port_full=initial_port_full,
                                            initial_vessel_empty=initial_vessel_empty,
                                            initial_vessel_full=initial_vessel_full
                                            )
            self._lp_action_logger.debug(f'{decision_event.tick},{self._port_idx2name[decision_event.port_idx]},{self._vessel_idx2name[decision_event.vessel_idx]},{decision_event.action_scope.discharge},{decision_event.action_scope.load},{action.quantity}')

            srandom.seed(decision_event.tick)

            _, decision_event, is_done = self._env.step(action)


        # Print summary
        self._print_summary(summary_name='LP Replay')

    def _print_summary(self, summary_name=None):
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
    runner = Runner(scenario=SCENARIO,
                    topology=TOPOLOGY,
                    max_tick=MAX_TICK,
                    )

    runner.start()
