
import numpy as np

from maro.simulator.core import Env
from scipy.sparse import coo_matrix
from maro.simulator.scenarios.citi_bike.common import DecisionType
from examples.citi_bike.ppo.action_shaping import ActionShaping

station_attribute_list = [
    'bikes', 'fulfillment', 'trip_requirement', 'shortage',
    'capacity', 'weekday', 'temperature', 'weather', 'holiday', 'min_bikes'
]

SHORTAGE_INDEX = 3
FULFILLMENT_INDEX = 1


class CitibikeStateShaping:
    def __init__(
        self,
        env: Env,
        td_steps=10,
        trip_threshold=2,
        feature_scaler=0.05,
        distance_scaler=0.05,
        time_scaler=0.05
    ):
        self._env = env
        self._td_steps = td_steps
        # self._inventory_idx = station_attribute_list.index('bikes')
        self._station_cnt = len(self._env.snapshot_list['stations'])
        self._trip_threshold = trip_threshold
        self._bool_identity = np.identity(self._station_cnt, dtype=np.bool)

        self.feature_scaler = feature_scaler
        self.distance_scaler = distance_scaler
        self.time_scaler = time_scaler
        self.action_scaler = ActionShaping.action_scaler
        print(f"*********** CitiBikeStateShaping ***********")

    @property
    def timestep(self):
        return self._td_steps

    def get_states(self, reward: object = None, decision_event: object = None, frame_index: int = -1):
        assert decision_event or frame_index >= 0, "valid frame_index should be provided if no decision event"
        cur_frame_index = decision_event.frame_index if decision_event else frame_index
        env_tick = self._env.tick if decision_event else self._env.get_frame_index_mapping()[frame_index][0]

        station_features, shortage, fulfillment = self.get_station_features(cur_frame_index, env_tick)
        edge_idx_list = self.extract_channels(cur_frame_index)

        acting_node_idx, action_edge_index, composed_amount = None, None, None
        if decision_event:
            acting_node_idx = decision_event.station_idx
            action_scope = decision_event.action_scope
            action_edge_index = np.array([[decision_event.station_idx]*len(action_scope), list(action_scope.keys())])
            neighbors = list(action_scope.keys())
            if decision_event.type == DecisionType.Supply:
                # get the remaining bikes of current bike station
                station_bikes = np.ones(len(neighbors)) * self._env.snapshot_list['stations'][
                    cur_frame_index: decision_event.station_idx: ['bikes']
                ][0]
                # get the capacity and bikes of neighbors
                tmp = self._env.snapshot_list['stations'][
                    cur_frame_index: neighbors: ['capacity', 'bikes']
                ]
                neighbor_docks = tmp[:len(neighbors)] - tmp[len(neighbors):]
                # get the legal amount each neighbor could receive
                legal_amount = np.min(np.vstack([neighbor_docks, station_bikes]), axis=0)
                composed_amount = self.action_scaler*np.hstack([legal_amount.reshape(-1, 1),
                                                                np.zeros((len(neighbors), 1))])
            else:
                ctmp, btmp = self._env.snapshot_list['stations'][
                    cur_frame_index: decision_event.station_idx: ['capacity', 'bikes']
                ]
                station_docks = np.ones(len(neighbors)) * (ctmp - btmp)
                neighbor_bikes = self._env.snapshot_list['stations'][cur_frame_index:neighbors:['bikes']]
                # get the legal amount each neighbor could supply
                legal_amount = np.min(np.vstack([neighbor_bikes, station_docks]), axis=0)
                composed_amount = self.action_scaler * np.hstack([np.zeros((len(neighbors), 1)),
                                                                  -legal_amount.reshape(-1, 1)])

        return {
            'acting_node_idx': acting_node_idx,
            'x': station_features,
            'edge_idx_list': edge_idx_list,
            'action_edge_idx': action_edge_index,
            'actual_amount': composed_amount,
            'node_cnt': self._station_cnt,
            'tick': cur_frame_index,
            'shortage': shortage,
            'fulfillment': fulfillment,
            }

    def get_station_features(self, cur_frame_index: int, env_tick: int):
        indexes = [cur_frame_index + bias for bias in range(0, -self._td_steps, -1)]

        raw_features = self.feature_scaler * self._env.snapshot_list['stations'][
            indexes:: station_attribute_list
        ].reshape(self._td_steps, -1, self._station_cnt).transpose((2, 1, 0))

        shortage = raw_features[:, SHORTAGE_INDEX, -1]
        fulfillment = raw_features[:, FULFILLMENT_INDEX, -1]

        station_features = np.concatenate(
            (
                raw_features.reshape(self._station_cnt, -1),
                np.ones((self._station_cnt, 1)) * (env_tick % 1440 / 1440.0),         # temporal offset in this day
                # np.ones((self._station_cnt, 1)) * (int(env_tick % 1440 / 60) / 24.0),  # info of hour
                np.identity(self._station_cnt)
            ),
            axis=1
        )

        return station_features, shortage, fulfillment

    def extract_channels(self, cur_frame_index: int):
        indexes = [cur_frame_index + bias for bias in range(0, -self._td_steps, -1)]

        raw_trips = self._env.snapshot_list['matrices'][
            indexes:: 'trips_adj'
        ].reshape(self._td_steps, self._station_cnt, self._station_cnt).transpose((1, 2, 0))

        # filter less frequent neighbors and add self loops
        adj = (np.sum(raw_trips, axis=2) > self._trip_threshold) | self._bool_identity
        sparse_adj = coo_matrix(adj)
        edge_index = np.stack((sparse_adj.row, sparse_adj.col), axis=0)
        return [edge_index, ]

    @property
    def channel_cnt(self):
        return 1

    @property
    def node_attr(self):
        return station_attribute_list

    @property
    def node_attr_len(self):
        return len(station_attribute_list)*self._td_steps+self._station_cnt+1
