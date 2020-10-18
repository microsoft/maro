# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np

from scipy.sparse import coo_matrix

from maro.simulator import Env
from maro.simulator.scenarios.citibike.common import DecisionType


class StateShaping:
    def __init__(self,
                 env: Env,
                 td_steps: int,
                 station_attribute_list: [str],
                 # transport_edge_index,
                 transport_time,
                 trip_threshold=10,
                 ):
        self._env = env
        self._td_steps = td_steps
        self._station_attribute_list = station_attribute_list
        self._inventory_idx = station_attribute_list.index('bikes')
        self._station_cnt = self._env.snapshot_list.static_node_number
        self._trip_threshold = trip_threshold
        self._time_mat = transport_time.reshape(*transport_time.shape, 1)
        # self._transport_edge_index = transport_edge_index
        self._bool_identity = np.identity(self._station_cnt, dtype=np.bool)

        self.feat_scaler = 0.05
        self.distance_scaler = 0.05
        self.time_scaler = 0.05
        self._time_mat *= self.time_scaler

    def __call__(self, decision_event):
        cur_tick = decision_event.frame_index
        ticks = [cur_tick + bias for bias in range(0, -self._td_steps, -1)]

        # cur_neighbor_idx_list = [0]*6
        station_feat_raw = self._env.snapshot_list.static_nodes[ticks:: (self._station_attribute_list, 0)].reshape(self._td_steps, self._station_cnt, len(self._station_attribute_list))
        station_feat_raw *= self.distance_scaler
        # id_feat = np.repeat((np.arange(self._station_cnt)/self._station_cnt).reshape(1, -1), self._td_steps, axis=0).reshape(self._td_steps, self._station_cnt, 1)
        # station_feat_raw = np.concatenate((station_feat_raw, id_feat), axis=-1)
        station_features = np.transpose(station_feat_raw, (1, 2, 0)).reshape(self._station_cnt, -1)
        id_feat = np.identity(self._station_cnt)
        station_features = np.concatenate((station_features, id_feat), axis=-1)

        trip_raw = self._env.snapshot_list.matrix[ticks: 'trip_adj'].reshape(self._td_steps, self._station_cnt,
                                                                             self._station_cnt)
        trip_adj = np.transpose(trip_raw, (1, 2, 0))
        distance_mat = self._env.snapshot_list.matrix[0:'distance_adj'].reshape(self._station_cnt, self._station_cnt,
                                                                                1) * self.distance_scaler
        # filter less frequent neighbors and add self loops
        adj = (np.sum(trip_adj, axis=2) > self._trip_threshold) | self._bool_identity
        sparse_adj = coo_matrix(adj)
        row, col = sparse_adj.row, sparse_adj.col
        edge_index = np.stack((row, col), axis=0)
        edge_x =  np.concatenate((trip_adj[row, col], distance_mat[row, col]), axis=1)

        '''
        # init_edge_index
        actual_amount = np.array(list(action_scope.values()))
        '''
        action_scope = decision_event.action_scope
        action_edge_index = np.array([[decision_event.cell_idx] * len(action_scope), list(action_scope.keys())])
        if decision_event.type == DecisionType.Supply:
            actual_amount = np.ones(len(decision_event.action_scope)) *\
                self._env.snapshot_list.static_nodes[cur_tick:decision_event.cell_idx:(['bikes'], 0)][0]
            actual_amount = np.hstack([actual_amount.reshape(-1, 1), np.zeros((*actual_amount.shape, 1))])
        else:
            ftmp, btmp = self._env.snapshot_list.static_nodes[cur_tick:decision_event.cell_idx:
                                                              (['capacity', 'bikes'], 0)]
            actual_amount = np.ones(len(decision_event.action_scope)) * (ftmp - btmp)
            actual_amount = np.hstack([np.zeros((*actual_amount.shape, 1)), -actual_amount.reshape(-1,1)])
        a_row, a_col = action_edge_index
        action_edge_x = self._time_mat[action_edge_index[0], action_edge_index[1]]
        # edge attribute: trip adj + transportation time
        action_edge_x = np.concatenate((trip_adj[a_row, a_col], action_edge_x), axis=1)

        return {
            'acting_node_idx': decision_event.cell_idx,
            # 'x': station_features.reshape(self._station_cnt, -1),
            'x': station_features,
            'edge_index': edge_index,
            'edge_x': edge_x,
            'action_edge_index': action_edge_index,
            'action_edge_x': action_edge_x,
            'actual_amount': actual_amount,
            'node_cnt': self._station_cnt,
            }
