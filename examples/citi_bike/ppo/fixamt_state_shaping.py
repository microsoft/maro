import numpy as np

from maro.simulator.core import Env
from examples.citi_bike.ppo.citibike_state_shaping import CitibikeStateShaping as BaseStateShaping
from examples.citi_bike.ppo.citibike_state_shaping import station_attribute_list, SHORTAGE_INDEX, FULFILLMENT_INDEX
from maro.simulator.utils.common import frame_index_to_ticks
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
        td_steps=10,
        trip_threshold=2,
        feature_scaler=0.05,
        distance_scaler=0.05,
        time_scaler=0.05,
        contains_id=False,
    ):
        super().__init__(env, td_steps, trip_threshold, feature_scaler, distance_scaler, time_scaler)
        self.contains_id = contains_id
        print(f"*********** CitiBikeStateShaping with Temporal State ***********")

    def get_station_features(self, cur_frame_index: int, env_tick: int):
        # indexes = [max(cur_frame_index + bias, 0) for bias in range(-self._td_steps+1, 1)]
        indexes = [max(cur_frame_index + bias, 0) for bias in range(0, -self._td_steps, -1)]
        idx2tick = frame_index_to_ticks(self._env._start_tick, self._env._start_tick + self._env._durations,
                                        self._env._snapshot_resolution)
        ticks = np.array([min(idx2tick[idx][-1], env_tick) for idx in indexes], dtype=np.int).reshape(-1, 1)

        raw_features = self.feature_scaler * self._env.snapshot_list['stations'][
            indexes:: station_attribute_list
        ].reshape(self._td_steps, self._station_cnt, len(station_attribute_list))
        raw_features = raw_features.transpose((0, 2, 1))
        shortage = raw_features[-1, SHORTAGE_INDEX, :]
        fulfillment = raw_features[-1, FULFILLMENT_INDEX, :]

        # add hour, week, month
        temporal_feature = np.concatenate((ticks // 1440 % 30, ticks // 1440 % 7, ticks % 1440 // 60),
                                          axis=1).reshape(self._td_steps, 1, 3)
        temporal_feature = np.repeat(temporal_feature, self._station_cnt, axis=1).reshape(self._td_steps,
                                                                                          self._station_cnt, 3)
        # change to (station_cnt, tick_len, attribute_len)
        station_features = np.concatenate((raw_features.transpose((2, 0, 1)), temporal_feature.transpose(1, 0, 2)),
                                          axis=-1)

        if self.contains_id:
            identity = np.repeat(np.identity(self._station_cnt).reshape(self._station_cnt, 1, self._station_cnt),
                                 self._td_steps, axis=1).reshape(self._station_cnt, self._td_steps, self._station_cnt)
            station_features = np.concatenate((station_features, identity), axis=-1)

        return station_features, shortage, fulfillment

    @property
    def channel_cnt(self):
        return 1

    @property
    def node_attr(self):
        return station_attribute_list

    @property
    def node_attr_len(self):
        base = len(station_attribute_list) + 3
        if self.contains_id:
            return base + self._station_cnt
        else:
            return base
