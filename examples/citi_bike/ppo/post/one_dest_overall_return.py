import numpy as np
import math
from collections import Iterable
from maro.simulator.scenarios.citibike.common import Action, DecisionEvent, DecisionType

class PostProcessor:
    def __init__(self, env, transfer_cost=0.02, gamma=0.9):
        self.env = env
        self.station_cnt = len(self.env.snapshot_list['stations'])
        self.trajectory = []
        self.last_decision_event = None
        self.cur_decision_event, self.cur_obs, self.cur_action = None, None, None
        max_cnt = 100
        self.gammas = np.logspace(0, 100, 100, base=gamma)
        self.transfer_cost = transfer_cost

        # scaler
        self.reward_scaler = 0.01

    def __call__(self):
        ret = np.zeros(self.station_cnt)
        valid_traj = self.trajectory[:-1]
        for i in range(len(valid_traj)-1, -1, -1):
            sars = valid_traj[i]
            ret = ret*sars['gamma'][0] + sars['r']
            sars['r'] = ret
        return valid_traj

    def normal_reward(self):
        decision_event, obs = self.cur_decision_event, self.cur_obs

        last = self.trajectory[-1]
        last['obs_'] = obs

        # reward computation
        order_data = self.env.snapshot_list['stations'][list(range(self.last_decision_event.frame_index, decision_event.frame_index+1)): :
                                                    ['fulfillment', 'shortage']].reshape(
                                                        decision_event.frame_index-self.last_decision_event.frame_index+1, self.station_cnt, 2)
        # reward_per_frame.shape: [frame_cnt, station_cnt]
        reward_per_frame = order_data[:,:,0] - order_data[:,:,1]
        # reward.shape: [station_cnt,]
        reward = self.gammas[:reward_per_frame.shape[0]].dot(reward_per_frame)
        '''
        # actions.shape: [action_cnt,]
        cur_action, cur_action_edge = last['a']
        # the transfer cost is appended on the received cell
        if self.last_decision_event.type == DecisionType.Supply:
            reward[cur_action_edge[1]] -= self.transfer_cost*cur_action
        else:
            reward[self.last_decision_event.station_idx] -= self.transfer_cost*np.sum(cur_action)
        '''
        last['r'] = reward*self.reward_scaler
        last['gamma'] = np.ones(reward.shape[0])*self.gammas[decision_event.frame_index-self.last_decision_event.frame_index]

        
    def record(self, decision_event=None, obs=None, action=None):
        if decision_event is not None and obs is not None:
            self.cur_decision_event = decision_event
            self.cur_obs = obs
        if action is not None:
            self.cur_action = action

        if self.cur_decision_event is None or self.cur_obs is None or self.cur_action is None:
            return

        if self.trajectory:
            self.normal_reward()
        # new exp recorded
        
        choice, amount, record_data = action
        
        self.trajectory.append({
            'obs': self.cur_obs,
            'a': np.array([choice, amount]),
            'obs_': None,
            'r': None,
            'gamma': np.ones(self.station_cnt),
            'supplement': record_data,
        })
        self.last_decision_event = self.cur_decision_event
        self.cur_decision_event, self.cur_obs, self.cur_action = None, None, None


    def reset(self):
        self.trajectory = []
        self.last_decision_event = None

