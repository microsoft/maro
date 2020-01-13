from datetime import datetime
from tqdm import tqdm

import io
import numpy as np
import os
import random
import torch
import yaml

from maro.simulator import Env
from maro.simulator.scenarios.ecr.common import Action, DecisionEvent
from maro.utils import convert_dottable
from maro.utils import Logger, LogFormat
from maro.utils import SimpleExperiencePool
from examples.ecr.common.action_shaping import ContinuousActionShaping, DiscreteActionShaping
from examples.ecr.common.state_shaping import StateShaping
from examples.ecr.ddpg.ddpg import Actor, Critic
from examples.ecr.demonstration.ddpg.demo_ddpg import DemoDDPG
from examples.ecr.demonstration.ddpg.demo_ddpg_agent import DemoDDPGAgent
from examples.ecr.demonstration.dqn.demo_dqn import DemoDQN
from examples.ecr.demonstration.dqn.demo_dqn_agent import DemoDQNAgent
from examples.ecr.demonstration.lp.demo_lp_agent import DemoLPAgent
from examples.ecr.demonstration.lp.reverse_action_shaping import ReverseActionShaping
from examples.ecr.demonstration.reinforce.demo_reinforce import DemoReinforce
from examples.ecr.demonstration.reinforce.demo_reinforce_agent import DemoReinforceAgent
from examples.ecr.online_lp.lp_action_shaping import LPActionShaping
from examples.ecr.online_lp.online_lp import Online_LP
from examples.ecr.q_learning.common.dqn import QNet
from examples.ecr.reinforce.reinforce import ActorNet

import maro.simulator.utils.random as srandom

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    srandom.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_config():
    # Parse config
    config_path = os.environ.get('CONFIG') or 'config.yml'
    with io.open(config_path, 'r') as in_file:
        raw_config = yaml.safe_load(in_file)
        config = convert_dottable(raw_config)

    log_folder = os.path.join(os.getcwd(), 'log', f"{datetime.now().strftime('%Y%m%d')}", config.experiment_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    with io.open(os.path.join(log_folder, 'config.yml'), 'w', encoding='utf8') as out_file:
        yaml.safe_dump(raw_config, out_file)

    return config, log_folder

class ActionRecorder:
    def __init__(self, episode_limit: int):
        self._max_episode = episode_limit
        self._chosen_action_sequence = []

    def __call__(self, current_episode: int, decision_event: DecisionEvent, action: Action):
        if current_episode < self._max_episode:
            self._chosen_action_sequence.append((decision_event, action))

    def get_event_action_sequence(self):
        return self._chosen_action_sequence

    def clear(self):
        self._chosen_action_sequence.clear()

class Runner:
    def __init__(self, log_folder: str, config):
        self._seed = config.seed
        self._log_folder = log_folder
        self._config = config
        
        self._init_env(config=config.env)
        self._action_recorder = ActionRecorder(episode_limit=config.demo.data_augmentation.num_episode)
        self._init_logger(log_folder=log_folder)

        # Load LP agent & RL agents
        self._lp_agent, self._rl_agent_dict = self._load_agents(agent_idx_list=self._env.agent_idx_list)

    def _init_env(self, config):
        set_seed(self._seed)
        self._env = Env(scenario=config.scenario,
                        topology=config.topology,
                        max_tick=config.max_tick)

        self._port_idx2name = self._env.node_name_mapping['static']
        self._vessel_idx2name = self._env.node_name_mapping['dynamic']
    
    def _init_logger(self, log_folder: str):
        self._logger = Logger(tag='runner', format_=LogFormat.none,
                              dump_folder=log_folder, dump_mode='w',
                              auto_timestamp=False, extension_name='txt')
        self._performance_logger = Logger(tag=f'runner.performance',
                                            dump_folder=log_folder, dump_mode='w', extension_name='csv',
                                            auto_timestamp=False)
        self._performance_logger.debug(
            f"episode,{','.join([port_name + '_booking' for port_name in self._port_idx2name.values()])},total_booking,{','.join([port_name + '_shortage' for port_name in self._port_idx2name.values()])},total_shortage")

    def _load_lp_agent(self, config, agent_idx_list: [int], state_shaping, shared_experience_pool_dict: dict):
        online_lp = Online_LP(port_idx2name = self._port_idx2name, 
                              vessel_idx2name=self._vessel_idx2name,
                              topo_config=self._env._business_engine.configs,
                              moving_average_length=config.forecasting.moving_window_size,
                              window_size=config.formulation.plan_window_size,
                              apply_buffer_length=config.formulation.apply_window_size,
                              time_decay=config.objective.time_decay_factor,
                              order_gain_factor=config.objective.order_gain_factor,
                              transit_cost_factor=config.objective.transit_cost_factor,
                              load_discharge_cost_factor=config.objective.load_discharge_cost_factor
                              )
        action_shaping = LPActionShaping()
        reverse_action_shaping = ReverseActionShaping(action_shaping=self._get_descrete_action_shaping())
        agent = DemoLPAgent(algorithm=online_lp,
                            demo_algorithm=self._config.demo.algorithm,
                            state_shaping=state_shaping,
                            action_shaping=action_shaping,
                            reverse_action_shaping=reverse_action_shaping,
                            reward_shaping=self._config.train.reward_shaping,
                            topology=self._config.env.topology,
                            port_idx2name=self._port_idx2name,
                            vessel_idx2name=self._vessel_idx2name,
                            rl_agent_idx_list=agent_idx_list,
                            experience_pool_dict=shared_experience_pool_dict,
                            log_folder=self._log_folder,
                            )
        return agent

    def _get_state_shaping(self):
        state_shaping = StateShaping(env=self._env,
                                     relative_tick_list=[-1, -2, -3, -4, -5, -6, -7],
                                     port_downstream_max_number=2,
                                     port_attribute_list=['empty', 'full', 'on_shipper', 'on_consignee', 'booking',
                                                          'shortage', 'fulfillment'],
                                     vessel_attribute_list=['empty', 'full', 'remaining_space'])
        return state_shaping
    
    def _get_descrete_action_shaping(self):
        discrete_action_num = self._config.train.discrete_action_num
        action_space = [round(i * 0.1, 1) for i in range(- (discrete_action_num // 2), discrete_action_num // 2 + 1)]
        action_shaping = DiscreteActionShaping(action_space=action_space)
        return action_shaping

    def _get_continuous_action_shaping(self):
        return ContinuousActionShaping()

    def _load_agents(self, agent_idx_list: [int]):
        rl_agent_dict = {}
        demo_experience_pool_dict = {}

        # Load RL agents, with a shared demo_experience pool
        state_shaping = self._get_state_shaping()
        if self._config.demo.algorithm == "ddpg":
            action_shaping = self._get_continuous_action_shaping()
        else:
            action_shaping = self._get_descrete_action_shaping()

        for agent_idx in agent_idx_list:
            experience_pool = SimpleExperiencePool()
            demo_experience_pool_dict[agent_idx] = SimpleExperiencePool()
            if self._config.demo.algorithm == 'dqn':
                rl_agent_dict[agent_idx] = self._init_dqn_agent(agent_idx=agent_idx,
                                                                state_shaping=state_shaping,
                                                                action_shaping=action_shaping,
                                                                self_experience_pool=experience_pool,
                                                                demo_experience_pool=demo_experience_pool_dict[agent_idx])
            elif self._config.demo.algorithm == 'reinforce':
                rl_agent_dict[agent_idx] = self._init_reinforce_agent(agent_idx=agent_idx,
                                                                state_shaping=state_shaping,
                                                                action_shaping=action_shaping,
                                                                self_experience_pool=experience_pool,
                                                                demo_experience_pool=demo_experience_pool_dict[agent_idx])
            elif self._config.demo.algorithm == 'ddpg':
                rl_agent_dict[agent_idx] = self._init_ddpg_agent(agent_idx=agent_idx,
                                                              state_shaping=state_shaping,
                                                              action_shaping=action_shaping,
                                                              self_experience_pool=experience_pool,
                                                              demo_experience_pool=demo_experience_pool_dict[agent_idx])
            else:
                print(f'Invalid demo algorithm: {self._config.demo.algorithm}!')
                exit()

        # Load LP agent, with the shared demo_experience_pool_dict
        lp_agent = self._load_lp_agent(config=self._config.online_lp,
                                       agent_idx_list=agent_idx_list,
                                       state_shaping=state_shaping,
                                       shared_experience_pool_dict=demo_experience_pool_dict)

        return lp_agent, rl_agent_dict

    def _init_dqn_agent(self, agent_idx, state_shaping, action_shaping, self_experience_pool, demo_experience_pool):
        set_seed(self._seed)
        policy_net = QNet(name=f'{self._port_idx2name[agent_idx]}.policy',
                          input_dim=state_shaping.dim,
                          hidden_dims=[256, 128, 64],
                          output_dim=self._config.train.discrete_action_num,
                          dropout_p=self._config.dqn.dropout_p,
                          log_enable=True,
                          log_folder=self._log_folder)
        target_net = QNet(name=f'{self._port_idx2name[agent_idx]}.target',
                          input_dim=state_shaping.dim,
                          hidden_dims=[256, 128, 64],
                          output_dim=self._config.train.discrete_action_num,
                          dropout_p=self._config.dqn.dropout_p,
                          log_enable=True,
                          log_folder=self._log_folder)
        target_net.load_state_dict(policy_net.state_dict())
        dqn = DemoDQN(policy_net=policy_net,
                  target_net=target_net,
                  gamma=self._config.dqn.gamma,
                  tau=self._config.dqn.tau,
                  target_update_frequency=self._config.dqn.target_update_frequency,
                  lr=self._config.train.learning_rate,
                  log_enable=True,
                  log_folder=self._log_folder,
                  log_dropout_p=0.95,
                  dashboard_enable=False,
                  dashboard=None)
        dqn_agent = DemoDQNAgent(agent_name=self._port_idx2name[agent_idx],
                                 topology=self._config.env.topology,
                                 port_idx2name=self._port_idx2name,
                                 vessel_idx2name=self._vessel_idx2name,
                                 algorithm=dqn,
                                 state_shaping=state_shaping,
                                 action_shaping=action_shaping,
                                 reward_shaping=self._config.train.reward_shaping,
                                 experience_pool=self_experience_pool,
                                 demo_experience_pool=demo_experience_pool,
                                 training_config=self._config.train,
                                 agent_idx_list=self._env.agent_idx_list,
                                 log_enable=True,
                                 log_folder=self._log_folder,
                                 dashboard_enable=False,
                                 dashboard=None)
        return dqn_agent

    def _init_reinforce_agent(self, agent_idx, state_shaping, action_shaping, self_experience_pool, demo_experience_pool):
        set_seed(self._seed)
        policy_net = ActorNet(name=f'{self._port_idx2name[agent_idx]}.policy',
                          input_dim=state_shaping.dim,
                          hidden_dims=[256, 128, 64],
                          output_dim=self._config.train.discrete_action_num,
                          log_enable=True,
                          log_folder=self._log_folder)

        reinforce = DemoReinforce(policy_net=policy_net,
                            lr=self._config.train.learning_rate,
                            log_enable=True,
                            log_folder=self._log_folder,
                            dashboard_enable=False,
                            dashboard=None)
        reinforce_agent = DemoReinforceAgent(agent_name=self._port_idx2name[agent_idx],
                                             topology=self._config.env.topology,
                                             port_idx2name=self._port_idx2name,
                                             vessel_idx2name=self._vessel_idx2name,
                                             algorithm=reinforce,
                                             state_shaping=state_shaping,
                                             action_shaping=action_shaping,
                                             reward_shaping=self._config.train.reward_shaping,
                                             experience_pool=self_experience_pool,
                                             demo_experience_pool=demo_experience_pool,
                                             training_config=self._config.train,
                                             agent_idx_list=self._env.agent_idx_list,
                                             agent_idx=agent_idx,
                                             log_enable=True,
                                             log_folder=self._log_folder,
                                             dashboard_enable=False,
                                             dashboard=None)
        return reinforce_agent

    def _init_ddpg_agent(self, agent_idx, state_shaping, action_shaping, self_experience_pool, demo_experience_pool):
        set_seed(self._seed)
        experience_pool = SimpleExperiencePool()
        actor_policy_net = Actor(name=f'{self._port_idx2name[agent_idx]}.policy_actor', 
                                    input_dim=state_shaping.dim,
                                    hidden_dims=[256, 128, 64], 
                                    output_dim=1, 
                                    dropout_actor=0,
                                    log_enable=True, 
                                    log_folder=self._log_folder)
        actor_target_net = Actor(name=f'{self._port_idx2name[agent_idx]}.target_actor', 
                                    input_dim=state_shaping.dim, 
                                    hidden_dims=[256, 128, 64], 
                                    output_dim=1, 
                                    dropout_actor=0,
                                    log_enable=True, 
                                    log_folder=self._log_folder)
        actor_target_net.load_state_dict(actor_policy_net.state_dict())

        critic_policy_net = Critic(name=f'{self._port_idx2name[agent_idx]}.policy_critic', 
                                    input_dim=state_shaping.dim,
                                    state_input_hidden_dims = [256, 128],
                                    action_input_hidden_dims = [128, 64, 32],
                                    action_dim=1, 
                                    dropout_critic=0,
                                    log_enable=True, 
                                    log_folder=self._log_folder)
        critic_target_net = Critic(name=f'{self._port_idx2name[agent_idx]}.target_critic', 
                                    input_dim=state_shaping.dim,
                                    state_input_hidden_dims = [256, 128],
                                    action_input_hidden_dims = [128, 64, 32],
                                    action_dim=1, 
                                    dropout_critic=0,
                                    log_enable=True,
                                    log_folder=self._log_folder)
        critic_target_net.load_state_dict(critic_policy_net.state_dict())

        ddpg = DemoDDPG(actor_policy_net=actor_policy_net, 
                        actor_target_net=actor_target_net,
                        critic_policy_net=critic_policy_net, 
                        critic_target_net=critic_target_net,
                        gamma=self._config.ddpg.gamma, 
                        tau=self._config.ddpg.tau,
                        target_update_frequency=self._config.ddpg.target_update_frequency,
                        critic_lr=self._config.ddpg.critic_lr, 
                        actor_lr=self._config.ddpg.actor_lr, 
                        sigma=self._config.ddpg.sigma, 
                        theta=self._config.ddpg.theta,
                        log_enable=True, 
                        log_folder=self._log_folder)
        
        return DemoDDPGAgent(agent_name=self._port_idx2name[agent_idx], 
                             topology=self._config.env.topology,
                             port_idx2name=self._port_idx2name,
                             vessel_idx2name=self._vessel_idx2name,
                             algorithm=ddpg,
                             experience_pool=experience_pool,
                             state_shaping=state_shaping,
                             action_shaping=action_shaping,
                             reward_shaping=self._config.train.reward_shaping,
                             training_config=self._config.train,
                             demo_experience_pool=demo_experience_pool,
                             agent_idx_list=self._env.agent_idx_list,
                             log_enable=True,
                             log_folder=self._log_folder)

    
    def _get_exploration_rate(self):
        total_episode = self._config.demo.total_episode
        rates = [0.0] * total_episode

        if not self._config.exploration.enable:
            return rates

        maximum_exploration_rate = self._config.exploration.max_exploration_rate
        first_phase_reduce_proportion = self._config.exploration.first_phase_reduce_proportion
        first_phase_proportion = self._config.exploration.phase_split_point

        first_phase_delta = maximum_exploration_rate * first_phase_reduce_proportion
        first_phase_episode_num = int(total_episode * first_phase_proportion)
        first_phase_step = first_phase_delta / max(first_phase_episode_num, 1)

        second_phase_delta = maximum_exploration_rate - first_phase_delta
        second_phase_episode_num = total_episode - first_phase_episode_num - 1
        second_phase_step = second_phase_delta / max(second_phase_episode_num, 1)

        rates[0] = maximum_exploration_rate
        for i in range(1, first_phase_episode_num):
            rates[i] = rates[i-1] - first_phase_step
        for i in range(first_phase_episode_num, total_episode - 1):
            rates[i] = rates[i-1] - second_phase_step
        
        return rates

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

    def _lp_data_augmentation(self, ep):
        max_tick = self._config.env.max_tick
        num_tick_between_plans = self._config.demo.data_augmentation.num_tick_between_plans
        interaction_window_size = self._config.demo.data_augmentation.interaction_window_size

        event_action_sequence = self._action_recorder.get_event_action_sequence()

        start_tick = 0
        while start_tick < max_tick:
            set_seed(ep)
            self._env.reset()

            _, decision_event, is_done = self._env.step(None)
            event_action_idx = 0
            while not is_done:
                if decision_event.tick < start_tick:
                    recorded_event, action = event_action_sequence[event_action_idx]
                    event_action_idx += 1
                    assert recorded_event.tick == decision_event.tick
                    assert recorded_event.port_idx == decision_event.port_idx
                    assert recorded_event.vessel_idx == decision_event.vessel_idx
                else:
                    initial_port_empty, initial_vessel_empty, initial_vessel_full = self._get_initial_values()
                    action = self._lp_agent.choose_action(decision_event=decision_event,
                                                          finished_events=self._env._event_buffer.get_finished_events(),
                                                          snapshot_list=self._env.snapshot_list,
                                                          initial_port_empty=initial_port_empty,
                                                          initial_vessel_empty=initial_vessel_empty,
                                                          initial_vessel_full=initial_vessel_full
                                                          )
                _, decision_event, is_done = self._env.step(action)
            
            self._print_summary(ep, summary_name=f'Online LP interaction from Tick {start_tick} | EP {ep}')

            # Save Experience
            # TODO: currently, interaction_window_size not used yet
            self._lp_agent.fulfill_cache(agent_idx_list=self._env.agent_idx_list,
                                         snapshot_list=self._env.snapshot_list,
                                         current_ep=ep)
            self._lp_agent.put_experience()
            self._lp_agent.reset()

            start_tick += num_tick_between_plans + 1

        self._action_recorder.clear()
        self._env.reset()

    def start(self):
        exploration_rates = self._get_exploration_rate()

        pbar = tqdm(range(self._config.demo.total_episode))
        for ep in pbar:
            set_seed(ep)
            pbar.set_description(f'Train Episode {ep}/{self._config.demo.total_episode}')

            _, decision_event, is_done = self._env.step(None)
            while not is_done:
                if self._config.demo.algorithm == "ddpg":
                    action = self._rl_agent_dict[decision_event.port_idx].choose_action(
                        decision_event=decision_event, is_test=False, current_ep=ep)
                else:
                    action = self._rl_agent_dict[decision_event.port_idx].choose_action(
                        decision_event=decision_event, eps=exploration_rates[ep], current_ep=ep)
                
                self._action_recorder(ep, decision_event, action)
                _, decision_event, is_done = self._env.step(action)

            # Fulfill reward and update experience pool
            for agent in self._rl_agent_dict.values():
                agent.fulfill_cache(agent_idx_list=self._env.agent_idx_list,
                                    snapshot_list=self._env.snapshot_list,
                                    current_ep=ep)
                agent.put_experience()
                agent.clear_cache()

            self._print_summary(ep=ep, summary_name=f'RL interaction EP {ep}')
            self._env.reset()
            
            # Demonstration Data Augmentation
            if ep < config.demo.data_augmentation.num_episode:
                self._lp_data_augmentation(ep)

            # Train models
            if not self._config.train.train_synchronously \
                or all([agent.meet_training_condition(ep) for agent in self._rl_agent_dict.values()]):
                for agent in self._rl_agent_dict.values():
                    agent.train(current_ep=ep)
                    # TODO: clear experience pool of reinforce

        # Save learned models of last episode
        for agent in self._rl_agent_dict.values():
            dump_path = f'{self._log_folder}/{agent._agent_name}_EP{ep}.pkl'
            agent.dump_policy_net_parameters(dump_path)


    def _print_summary(self, ep, summary_name=None):
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
    config, log_folder = parse_config()

    # Initialize and start runner
    runner = Runner(log_folder=log_folder, config=config)
    runner.start()