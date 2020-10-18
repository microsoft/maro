import numpy as np
from collections import Iterable
from maro.simulator.scenarios.citibike.common import DecisionType


class PostProcessor:
    def __init__(self, env, transfer_cost=0.02, gamma=0.5):
        self.env = env
        self.station_cnt = len(self.env.snapshot_list['stations'])
        self.trajectory = []
        self.last_decision_event = None
        self.cur_decision_event, self.cur_obs, self.cur_action = None, None, None
        self.gammas = np.logspace(0, 100, 100, base=gamma)
        self.transfer_cost = transfer_cost

        # scaler
        self.reward_scaler = 0.05

    def __call__(self):
        return self.trajectory[:-1]

    def cal_reward(self):
        decision_event, obs = self.cur_decision_event, self.cur_obs
        # last_cell_idxes = list(self.last_decision_event.action_scope.keys())
        last = self.trajectory[-1]
        last['obs_'] = obs
        cur_action, cur_action_edge = last['a']
        last_cell_idxes = cur_action_edge[1]
        last_frame_idx = min(max(0, decision_event.frame_index-1), self.last_decision_event.frame_index)
        tmp = self.env.snapshot_list['stations'][list(range(last_frame_idx, decision_event.frame_index+1))
                                                    :list(last_cell_idxes):['shortage', 'bikes']]
        tmp = tmp.reshape(-1, len(last_cell_idxes), 2)
        shortage, bikes = tmp[:,:,0], tmp[:,:,1]
        shortage = np.sum(shortage, axis=0)
        bikes = np.mean(bikes, axis=0)
        if self.last_decision_event.type == DecisionType.Supply:
            # reward = (shortage != 0)*(cur_action-1) + (shortage == 0)*(-cur_action*0.2)
            reward = (shortage + 2 - 0.5*bikes) * (-cur_action)
            # reward = np.vstack((shortage, cur_action)).min(axis=0)
            last['r'] = self.reward_scaler * reward
        else:
            # reward = (cur_action - inventory)*(1+shortage//(shortage+fulfillment+0.001))
            reward = (shortage + 2 - 0.5*bikes) * (cur_action)
            last['r'] = self.reward_scaler * reward

        last['gamma'] = np.ones(reward.shape[0])*self.gammas[decision_event.frame_index-last_frame_idx]

    def normal_reward(self):
        decision_event, obs = self.cur_decision_event, self.cur_obs

        last = self.trajectory[-1]
        last['obs_'] = obs

        # reward computation
        order_data = self.env.snapshot_list['stations'][list(range(self.last_decision_event.frame_index,
                                                        decision_event.frame_index+1))::
                                                        ['fulfillment', 'shortage']].reshape(decision_event.frame_index
                                                        - self.last_decision_event.frame_index+1, self.station_cnt, 2)
        # reward_per_frame.shape: [frame_cnt, station_cnt]
        reward_per_frame = order_data[:, :, 0] - order_data[:, :, 1]
        # reward.shape: [station_cnt,]
        reward = self.gammas[:reward_per_frame.shape[0]].dot(reward_per_frame)
        # actions.shape: [action_cnt,]
        cur_action, cur_action_edge = last['a']
        # the transfer cost is appended on the received cell
        if self.last_decision_event.type == DecisionType.Supply:
            reward[cur_action_edge[1]] -= self.transfer_cost*cur_action
        else:
            reward[self.last_decision_event.station_idx] -= self.transfer_cost*np.sum(cur_action)
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

        if isinstance(self.cur_action, Iterable):
            action_edge = np.array([[a.from_station_idx for a in self.cur_action], [a.to_station_idx for a in self.cur_action]],
                            np.int)
            action_amount = np.array([a.number for a in self.cur_action])
        else:
            action_edge = np.array([[self.cur_action.from_station_idx], [self.cur_action.to_station_idx]], np.int)
            action_amount = np.array([self.cur_action.number])

        self.trajectory.append({
            'obs': self.cur_obs,
            'a': (action_amount, action_edge),
            'obs_': None,
            'r': None,
            'gamma': np.ones(self.station_cnt),
        })
        self.last_decision_event = self.cur_decision_event
        self.cur_decision_event, self.cur_obs, self.cur_action = None, None, None

    '''
    def __call__(self):
        # do not return the last, because the last one is not complete.
        # because we aim to solve continuous problem, there is no episode end.
        # rewards = [e['r'] for e in self.trajectory]
        # gamma = [e['gamma'] for e in self.trajectory]
        # returns = np.zeros((len(rewards), self.station_cnt))
        # for i in range(len(rewards)-2, -1, -1):
        #     returns[i] = rewards[i] + gamma[i+1]*returns[i+1]
        #     self.trajectory[i]['return'] = returns[i]
        return self.trajectory[:-1]
    '''

    def reset(self):
        self.trajectory = []
        self.last_decision_event = None
