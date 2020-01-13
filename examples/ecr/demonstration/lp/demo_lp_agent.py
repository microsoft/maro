import torch

from examples.ecr.common.reward_shaping import GoldenFingerReward, GoldenFingerRewardContinuous, TruncateReward, SelfAwareTruncatedReward
from examples.ecr.online_lp.lp_agent import LPAgent
from maro.simulator.scenarios.ecr.common import Action, DecisionEvent
from maro.utils import Logger, LogFormat


class DemoLPAgent(LPAgent):
    def __init__(self,
                 algorithm,
                 demo_algorithm,
                 state_shaping,
                 action_shaping,
                 reverse_action_shaping,
                 reward_shaping: str,
                 topology: str,
                 port_idx2name: dict,
                 vessel_idx2name: dict,
                 rl_agent_idx_list: [int],
                 experience_pool_dict: dict,
                 log_folder: str,
                 ):
        super(DemoLPAgent, self).__init__(algorithm=algorithm,
                                          action_shaping=action_shaping,
                                          port_idx2name=port_idx2name,
                                          vessel_idx2name=vessel_idx2name)
        self._demo_algorithm = demo_algorithm
        self._state_shaping = state_shaping
        self._reverse_action_shaping = reverse_action_shaping
        self._rl_agent_idx_list = rl_agent_idx_list
        self._experience_pool_dict = experience_pool_dict
        # Initialize reward shaping
        self._reward_shaping_dict = {}
        for agent_idx in self._rl_agent_idx_list:
            if reward_shaping == 'gf':
                self._reward_shaping_dict[agent_idx] = GoldenFingerReward(topology=topology, action_space=self._reverse_action_shaping.action_space, base=10)
            elif reward_shaping == 'tc':
                self._reward_shaping_dict[agent_idx] = TruncateReward(agent_idx_list=self._rl_agent_idx_list)
            elif reward_shaping == 'cgf':
                self._reward_shaping_dict[agent_idx] = GoldenFingerRewardContinuous(topology=topology, base=10)
            elif reward_shaping == 'stc':
                self._reward_shaping_dict[agent_idx] = SelfAwareTruncatedReward(agent_idx_list=self._rl_agent_idx_list, agent_idx=agent_idx)
            else:
                raise KeyError(f'Invalid reward shaping: {reward_shaping}')
        # Initialize experience cache
        self._action_tick_cache = {}
        self._state_cache = {}
        self._action_cache = {}
        self._actual_action_cache = {}
        self._reward_cache = {}
        self._next_state_cache = {}
        self._decision_event_cache = {}
        self._port_states_cache = {}
        self._vessel_states_cache = {}
        for agent_idx in self._rl_agent_idx_list:
            self._action_tick_cache[agent_idx] = []
            self._state_cache[agent_idx] = []
            self._action_cache[agent_idx] = []
            self._actual_action_cache[agent_idx] = []
            self._reward_cache[agent_idx] = []
            self._next_state_cache[agent_idx] = []
            self._decision_event_cache[agent_idx] = []
            self._port_states_cache[agent_idx] = []
            self._vessel_states_cache[agent_idx] = []
        # Initialize logger
        self._choose_action_logger_dict = {}
        for agent_idx in self._rl_agent_idx_list:
            self._choose_action_logger_dict[agent_idx] = Logger(tag=f'demo.{self._port_idx2name[agent_idx]}.choose_action',
                                                                format_=LogFormat.none,
                                                                dump_folder=log_folder, dump_mode='w', extension_name='csv',
                                                                auto_timestamp=False)
            self._choose_action_logger_dict[agent_idx].debug(
                'episode,tick,port_empty,port_full,port_on_shipper,port_on_consignee,vessel_empty,vessel_full,vessel_remaining_space,max_load_num,max_discharge_num,vessel_name,action_index,actual_action,reward'
            )

    def choose_action(self,
                      decision_event: DecisionEvent,
                      finished_events: list,
                      snapshot_list: list,
                      initial_port_empty: dict = None,
                      initial_vessel_empty: dict = None,
                      initial_vessel_full: dict = None,
                      ) -> Action:
        cur_tick = decision_event.tick
        cur_port_idx = decision_event.port_idx
        cur_vessel_idx = decision_event.vessel_idx
        env_action = super(DemoLPAgent, self).choose_action(decision_event=decision_event,
                                                            finished_events=finished_events,
                                                            snapshot_list=snapshot_list,
                                                            initial_port_empty=initial_port_empty,
                                                            initial_vessel_empty=initial_vessel_empty,
                                                            initial_vessel_full=initial_vessel_full)
        # Generate and cache experience data
        numpy_state = self._state_shaping(cur_tick=cur_tick, cur_port_idx=cur_port_idx, cur_vessel_idx=cur_vessel_idx)
        state = torch.from_numpy(numpy_state).view(1, len(numpy_state))
        self._state_cache[cur_port_idx].append(numpy_state)

        actual_action = env_action.quantity
        self._actual_action_cache[cur_port_idx].append(actual_action)
        if self._demo_algorithm == "ddpg":
            self._action_cache[cur_port_idx].append(actual_action)
        else:
            action_index = self._reverse_action_shaping(scope=decision_event.action_scope, env_action=actual_action)
            self._action_cache[cur_port_idx].append(action_index)
        self._action_tick_cache[cur_port_idx].append(cur_tick)

        self._decision_event_cache[cur_port_idx].append(decision_event)
        port_states = snapshot_list.static_nodes[
                      cur_tick: [cur_port_idx]: (['empty', 'full', 'on_shipper', 'on_consignee'], 0)]
        vessel_states = snapshot_list.dynamic_nodes[
                        cur_tick: [cur_vessel_idx]: (['empty', 'full', 'remaining_space'], 0)]
        self._port_states_cache[cur_port_idx].append(port_states)
        self._vessel_states_cache[cur_port_idx].append(vessel_states)

        return env_action

    def _fulfill_single_cache(self, agent_idx, agent_idx_list: [int], snapshot_list, current_ep: int):
        for i, tick in enumerate(self._action_tick_cache[agent_idx]):
            if type(self._reward_shaping_dict[agent_idx]) == GoldenFingerReward:
                self._reward_shaping_dict[agent_idx](port_name=self._port_idx2name[self._decision_event_cache[agent_idx][i].port_idx],
                                                     vessel_name=self._vessel_idx2name[self._decision_event_cache[agent_idx][i].vessel_idx],
                                                     action_index=self._action_cache[agent_idx][i])
            elif type(self._reward_shaping_dict[agent_idx]) == GoldenFingerRewardContinuous:
                self._reward_shaping_dict[agent_idx](port_name=self._port_idx2name[self._decision_event_cache[agent_idx][i].port_idx],
                                                     vessel_name=self._vessel_idx2name[self._decision_event_cache[agent_idx][i].vessel_idx],
                                                     action_value=self._action_cache[agent_idx][i])
            elif type(self._reward_shaping_dict[agent_idx]) == TruncateReward:
                self._reward_shaping_dict[agent_idx](snapshot_list=snapshot_list, start_tick=tick + 1, end_tick=tick + 100)
            elif type(self._reward_shaping_dict[agent_idx]) == SelfAwareTruncatedReward:
                self._reward_shaping_dict[agent_idx](snapshot_list=snapshot_list, start_tick=tick + 1, end_tick=tick + 100)
            else:
                raise KeyError(f'Unknown reward shaping type: {type(self._reward_shaping_dict[agent_idx])}')

        self._reward_cache[agent_idx] = self._reward_shaping_dict[agent_idx].reward_cache
        self._action_tick_cache[agent_idx] = []
        self._next_state_cache[agent_idx] = self._state_cache[agent_idx][1:]
        self._state_cache[agent_idx] = self._state_cache[agent_idx][:-1]
        self._action_cache[agent_idx] = self._action_cache[agent_idx][:-1]
        self._actual_action_cache[agent_idx] = self._actual_action_cache[agent_idx][:-1]
        self._decision_event_cache[agent_idx] = self._decision_event_cache[agent_idx][:-1]
        self._port_states_cache[agent_idx] = self._port_states_cache[agent_idx][:-1]
        self._vessel_states_cache[agent_idx] = self._vessel_states_cache[agent_idx][:-1]

        for i, decision_event in enumerate(self._decision_event_cache[agent_idx]):
            episode = current_ep
            tick = decision_event.tick
            port_states = self._port_states_cache[agent_idx][i]
            vessel_states = self._vessel_states_cache[agent_idx][i]
            max_load_num = decision_event.action_scope.load
            max_discharge_num = decision_event.action_scope.discharge
            vessel_name = self._vessel_idx2name[decision_event.vessel_idx]
            action_index = self._action_cache[agent_idx][i]
            actual_action = self._actual_action_cache[agent_idx][i]
            reward = self._reward_cache[agent_idx][i]
            log_str = f"{episode},{tick},{','.join([str(f) for f in port_states])},{','.join([str(f) for f in vessel_states])},{max_load_num},{max_discharge_num},{vessel_name},{action_index},{actual_action},{reward}"
            self._choose_action_logger_dict[agent_idx].debug(log_str)

    def fulfill_cache(self, agent_idx_list: [int], snapshot_list, current_ep: int):
        for agent_idx in self._rl_agent_idx_list:
            self._fulfill_single_cache(agent_idx, agent_idx_list, snapshot_list, current_ep)

    def _put_single_port_experience(self, agent_idx):
        self._experience_pool_dict[agent_idx].put(category_data_batches=[
            ('state', self._state_cache[agent_idx]),
            ('action', self._action_cache[agent_idx]),
            ('reward', self._reward_cache[agent_idx]),
            ('next_state', self._next_state_cache[agent_idx]),
            ('actual_action', self._actual_action_cache[agent_idx])
        ])

    def put_experience(self):
        for agent_idx in self._rl_agent_idx_list:
            self._put_single_port_experience(agent_idx)

    def reset(self):
        super(DemoLPAgent, self).reset()
        # Clear cache
        for agent_idx in self._rl_agent_idx_list:
            # TODO: confirm interface
            self._reward_shaping_dict[agent_idx].clear_cache()
            self._action_tick_cache[agent_idx] = []
            self._state_cache[agent_idx] = []
            self._action_cache[agent_idx] = []
            self._actual_action_cache[agent_idx] = []
            self._reward_cache[agent_idx] = []
            self._next_state_cache[agent_idx] = []
            self._decision_event_cache[agent_idx] = []
            self._port_states_cache[agent_idx] = []
            self._vessel_states_cache[agent_idx] = []