import numpy as np

from maro.simulator.core import Env
from scipy.sparse import coo_matrix
from maro.simulator.scenarios.citi_bike.common import DecisionType
from examples.citi_bike.ppo.action_shaping import ActionShaping
from examples.citi_bike.ppo.citibike_state_shaping import CitibikeStateShaping as BaseStateShaping
from examples.citi_bike.ppo.citibike_state_shaping import station_attribute_list, SHORTAGE_INDEX, FULFILLMENT_INDEX
from maro.simulator.utils.common import tick_to_frame_index,frame_index_to_ticks
import pickle

'''
station_attribute_list = [
    'bikes', 'fulfillment', 'trip_requirement', 'shortage',
    'capacity', 'weekday', 'temperature', 'weather', 'holiday'
]
SHORTAGE_INDEX = 3
FULFILLMENT_INDEX = 1
'''



class CitibikeStateShaping(BaseStateShaping):
    def __init__(
        self,
        env: Env,
        td_steps=20,
        trip_threshold=2,
        feature_scaler=0.05,
        distance_scaler=0.05,
        time_scaler=0.05,
        contains_id=False,
    ):
        super().__init__(env, td_steps, trip_threshold, feature_scaler, distance_scaler, time_scaler)
        self.contains_id = contains_id
        with open(r'data/citibike/201901/filtered_edge_list.bin', 'rb') as fp:
            self.edge_list = pickle.load(fp)
        print(f"*********** CitiBikeStateShaping with Temporal State ***********")

    def get_states(self, reward: object=None, decision_event: object=None, frame_index: int=-1):
        assert decision_event or frame_index >= 0, "valid frame_index should be provided if no decision event"
        cur_frame_index = decision_event.frame_index if decision_event else frame_index
        env_tick = self._env.tick if decision_event else self._env.get_frame_index_mapping()[frame_index][0]

        station_features, shortage, fulfillment = self.get_station_features(cur_frame_index, env_tick)
        edge_idx_list = self.extract_channels(cur_frame_index)

        acting_node_idx, action_edge_index, legal_amount = None, None, None
        if decision_event:
            acting_node_idx = decision_event.station_idx
            neighbors = [k for k in decision_event.action_scope.keys() if k != decision_event.station_idx]
            action_edge_index = np.array([[decision_event.station_idx,]*len(neighbors), neighbors])
            if decision_event.type == DecisionType.Supply:
                # get the remaining bikes of current bike station
                station_bikes = np.ones(len(neighbors)) * self._env.snapshot_list['stations'][
                    cur_frame_index : decision_event.station_idx : ['bikes']
                ][0]
                # get the capacity and bikes of neighbors
                tmp = self._env.snapshot_list['stations'][
                    cur_frame_index : neighbors : ['capacity', 'bikes']
                ]
                neighbor_docks = tmp[:len(neighbors)] - tmp[len(neighbors):]
                # get the legal amount each neighbor could receive
                legal_amount = np.min(np.vstack([neighbor_docks, station_bikes]), axis=0)*self.action_scaler
            else:
                ctmp, btmp = self._env.snapshot_list['stations'][
                    cur_frame_index : decision_event.station_idx : ['capacity', 'bikes']
                ]
                station_docks = np.ones(len(neighbors)) * (ctmp - btmp)
                neighbor_bikes = self._env.snapshot_list['stations'][cur_frame_index:neighbors:['bikes']]
                # get the legal amount each neighbor could supply
                legal_amount = - np.min(np.vstack([neighbor_bikes, station_docks]), axis=0)*self.action_scaler
        
        return {
            'acting_node_idx': np.array([acting_node_idx]),
            'x': station_features,
            'edge_idx_list': edge_idx_list, 
            'action_edge_idx': action_edge_index, 
            'actual_amount': legal_amount,
            'node_cnt': self._station_cnt,
            'tick': cur_frame_index,
            'shortage': shortage,
            'fulfillment': fulfillment,
            }


    # def get_station_features(self, cur_frame_index: int, env_tick: int):
    #     indexes = [cur_frame_index + bias for bias in range(0, -self._td_steps, -1)]

    #     raw_features = self.feature_scaler * self._env.snapshot_list['stations'][
    #         indexes : : station_attribute_list
    #     ].reshape(self._td_steps, -1, self._station_cnt).transpose((2, 1, 0))

    #     shortage = raw_features[:, SHORTAGE_INDEX, -1]
    #     fulfillment = raw_features[:, FULFILLMENT_INDEX, -1]
        
    #     station_features = np.concatenate(
    #         (
    #             raw_features.reshape(self._station_cnt, -1),
    #             np.ones((self._station_cnt, 1)) * (env_tick % 1440 / 1440.0),         # temporal offset in this day
    #             # np.ones((self._station_cnt, 1)) * (int(env_tick % 1440 / 60) / 24.0),  # info of hour
    #             np.identity(self._station_cnt)
    #         ),
    #         axis=1
    #     )

    #     return station_features, shortage, fulfillment

    def get_station_features(self, cur_frame_index: int, env_tick: int):
        indexes = [max(cur_frame_index + bias, 0) for bias in range(-self._td_steps+1, 1)]
        # indexes = [max(cur_frame_index + bias, 0) for bias in range(0,-self._td_steps, -1)]
        idx2tick = frame_index_to_ticks(self._env._start_tick, self._env._start_tick+self._env._durations, self._env._snapshot_resolution)
        ticks = np.array([min(idx2tick[idx][-1], env_tick) for idx in indexes], dtype=np.int).reshape(-1, 1)

        raw_features = self.feature_scaler * self._env.snapshot_list['stations'][
            indexes : : station_attribute_list
        ].reshape(self._td_steps, len(station_attribute_list), self._station_cnt)
        shortage = raw_features[-1, SHORTAGE_INDEX, :]
        fulfillment = raw_features[-1, FULFILLMENT_INDEX, :]
        season = np.zeros((self._td_steps,1))
        month = np.zeros((self._td_steps,1))
        week = np.zeros((self._td_steps,1))
        dw = ticks//1440%7
        hd = ticks%1440//60
        mh = ticks%1440%60//20
        time_feature = np.concatenate((season,month,week,dw,hd,mh),axis=1).reshape(self._td_steps,1,6)
        time_feature = np.repeat(time_feature,self._station_cnt,axis=1).reshape(self._td_steps,self._station_cnt,6)
        # change to (station_cnt, tick_len, attribute_len)
        station_features = raw_features.transpose((0, 2, 1))

        if self.contains_id:
            identity = np.repeat(np.identity(self._station_cnt).reshape(self._station_cnt, 1, self._station_cnt), self._td_steps, axis=1).reshape(self._station_cnt, self._td_steps, self._station_cnt)
            station_features = np.concatenate((station_features, identity), axis=-1)
        
        return (station_features, time_feature), shortage, fulfillment

    
    def extract_channels(self, cur_frame_index: int):
        return [self.edge_list]

    @property
    def channel_cnt(self):
        return 1

    @property
    def node_attr(self):
        return station_attribute_list

    @property
    def node_attr_len(self):
        base = len(station_attribute_list)
        if self.contains_id:
            return base + self._station_cnt
        else:
            return base
