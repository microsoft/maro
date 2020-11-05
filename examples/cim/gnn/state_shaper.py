import numpy as np

from maro.rl.shaping.state_shaper import StateShaper

from .utils import compute_v2p_degree_matrix


class GNNStateShaper(StateShaper):
    """State shaper to extract graph information.

    Args:
        port_code_list (list): The list of the port codes in the CIM topology.
        vessel_code_list (list): The list of the vessel code in the CIM topology.
        max_tick (int): The duration of the simulation.
        feature_config (dict): The dottable dict that stores the configuration of the observation feature.
        max_value (int): The norm scale. All the feature are simply divided by this number.
        tick_buffer (int): The value n in n-step TD.
        only_demo (bool): Define if the shaper instance is used only for shape demonstration(True) or runtime
            shaping(False).
    """

    def __init__(
            self, port_code_list, vessel_code_list, max_tick, feature_config, max_value=100000, tick_buffer=20,
            only_demo=False):
        # Collect and encode all ports.
        self.port_code_list = list(port_code_list)
        self.port_cnt = len(self.port_code_list)
        self.port_code_inv_dict = {code: i for i, code in enumerate(self.port_code_list)}

        # Collect and encode all vessels.
        self.vessel_code_list = list(vessel_code_list)
        self.vessel_cnt = len(self.vessel_code_list)
        self.vessel_code_inv_dict = {code: i for i, code in enumerate(self.vessel_code_list)}

        # Collect and encode ports and vessels together.
        self.node_code_inv_dict_p = {i: i for i in self.port_code_list}
        self.node_code_inv_dict_v = {i: i + self.port_cnt for i in self.vessel_code_list}
        self.node_cnt = self.port_cnt + self.vessel_cnt

        one_hot_coding = np.identity(self.node_cnt)
        self.port_one_hot_coding = np.expand_dims(one_hot_coding[:self.port_cnt], axis=0)
        self.vessel_one_hot_coding = np.expand_dims(one_hot_coding[self.port_cnt:], axis=0)
        self.last_tick = -1

        self.port_features = [
            "empty", "full", "capacity", "on_shipper", "on_consignee", "booking", "acc_booking", "shortage",
            "acc_shortage", "fulfillment", "acc_fulfillment"]
        self.vessel_features = ["empty", "full", "capacity", "remaining_space"]

        self._max_tick = max_tick
        self._tick_buffer = tick_buffer
        # To identify one vessel would never arrive at the port.
        self.max_arrival_time = 99999999

        self.vedge_dim = 2
        self.pedge_dim = 1

        self._only_demo = only_demo
        self._feature_config = feature_config
        self._normalize = True
        self._norm_scale = 2.0 / max_value
        if not only_demo:
            self._state_dict = {
                # Last "tick" is used for embedding, all zero and never be modified.
                "v": np.zeros((self._max_tick + 1, self.vessel_cnt, self.get_input_dim("v"))),
                "p": np.zeros((self._max_tick + 1, self.port_cnt, self.get_input_dim("p"))),
                "vo": np.zeros((self._max_tick + 1, self.vessel_cnt, self.port_cnt), dtype=np.int),
                "po": np.zeros((self._max_tick + 1, self.port_cnt, self.vessel_cnt), dtype=np.int),
                "vedge": np.zeros((self._max_tick + 1, self.vessel_cnt, self.port_cnt, self.get_input_dim("vedge"))),
                "pedge": np.zeros((self._max_tick + 1, self.port_cnt, self.vessel_cnt, self.get_input_dim("vedge"))),
                "ppedge": np.zeros((self._max_tick + 1, self.port_cnt, self.port_cnt, self.get_input_dim("pedge"))),
            }

            # Fixed order: in the order of degree.

    def compute_static_graph_structure(self, env):
        v2p_adj_matrix = compute_v2p_degree_matrix(env)
        p2p_adj_matrix = np.dot(v2p_adj_matrix.T, v2p_adj_matrix)
        p2p_adj_matrix[p2p_adj_matrix == 0] = self.max_arrival_time
        np.fill_diagonal(p2p_adj_matrix, self.max_arrival_time)
        self._p2p_embedding = self.sort(p2p_adj_matrix)

        v2p_adj_matrix = -v2p_adj_matrix
        v2p_adj_matrix[v2p_adj_matrix == 0] = self.max_arrival_time
        self._fixed_v_order = self.sort(v2p_adj_matrix)
        self._fixed_p_order = self.sort(v2p_adj_matrix.T)

    @property
    def p2p_static_graph(self):
        return self._p2p_embedding

    def sort(self, arrival_time, attr=None):
        """
        Given the arrival time matrix, this function sort the matrix and return the index matrix in the order of
        arrival time
        """
        n, m = arrival_time.shape
        if self._feature_config.attention_order == "ramdom":
            arrival_time = arrival_time + np.random.randint(self._max_tick, size=arrival_time.shape)
        at_index = np.argsort(arrival_time, axis=1)
        if attr is not None:
            idx_tmp = np.repeat(at_index, attr.shape[-1]).reshape(*at_index.shape, attr.shape[-1])
            attr = np.take_along_axis(attr, idx_tmp, axis=1)
        mask = np.sort(arrival_time, axis=1) >= self.max_arrival_time
        at_index += 1
        at_index[mask] = 0
        if attr is None:
            return at_index
        else:
            return at_index, attr

    def end_ep_callback(self, snapshot_list):
        if self._only_demo:
            return
        tick_range = np.arange(start=self.last_tick, stop=self._max_tick)
        self._sync_raw_features(snapshot_list, list(tick_range))
        self.last_tick = -1

    def _sync_raw_features(self, snapshot_list, tick_range, static_code=None, dynamic_code=None):
        """This function update the state_dict from snapshot_list in the given tick_range."""
        if len(tick_range) == 0:
            # This occurs when two actions happen at the same tick.
            return

        # One dim features.
        port_naive_feature = snapshot_list["ports"][tick_range: self.port_code_list: self.port_features] \
            .reshape(len(tick_range), self.port_cnt, -1)
        # Number of laden from source to destination.
        full_on_port = snapshot_list["matrices"][tick_range::"full_on_ports"].reshape(
            len(tick_range), self.port_cnt, self.port_cnt)
        # Normalize features to a small range.
        port_state_mat = self.normalize(port_naive_feature)

        if self._feature_config.onehot_identity:
            # Add onehot vector to identify port and vessel.
            port_onehot = np.repeat(self.port_one_hot_coding, len(tick_range), axis=0)
            if static_code is not None and dynamic_code is not None:
                # Identify the decision vessel at the decision port.
                port_onehot[-1, self.port_code_inv_dict[static_code], self.node_code_inv_dict_v[dynamic_code]] = -1
            port_state_mat = np.concatenate([port_state_mat, port_onehot], axis=2)
        self._state_dict["p"][tick_range] = port_state_mat

        vessel_naive_feature = snapshot_list["vessels"][tick_range:self.vessel_code_list: self.vessel_features] \
            .reshape(len(tick_range), self.vessel_cnt, -1)
        full_on_vessel = snapshot_list["matrices"][tick_range::"full_on_vessels"].reshape(
            len(tick_range), self.vessel_cnt, self.port_cnt)

        vessel_state_mat = self.normalize(vessel_naive_feature)
        if self._feature_config.onehot_identity:
            vessel_state_mat = np.concatenate(
                [vessel_state_mat, np.repeat(self.vessel_one_hot_coding, len(tick_range), axis=0)], axis=2)
        self._state_dict["v"][tick_range] = vessel_state_mat

        # last_arrival_time.shape: vessel_cnt * port_cnt
        # -1 means one vessel never stops at the port
        vessel_arrival_time = snapshot_list["matrices"][tick_range[-1]:: "vessel_plans"].reshape(
            self.vessel_cnt, self.port_cnt)
        # Use infinity time to identify vessels never arrive at the port.
        last_arrival_time = vessel_arrival_time + 1
        last_arrival_time[last_arrival_time == 0] = self.max_arrival_time
        if static_code is not None and dynamic_code is not None:
            # To differentiate vessel acting on the port and other vessels that have taken or wait to take actions.
            last_arrival_time[self.vessel_code_inv_dict[dynamic_code], self.port_code_inv_dict[static_code]] = 0

        # Here, we assume that the order of arriving time between two action/event is all the same.
        vedge_raw = self.normalize(np.stack((full_on_vessel[-1], last_arrival_time), axis=-1))
        vo, vedge = self.sort(last_arrival_time, attr=vedge_raw)
        po, pedge = self.sort(last_arrival_time.T, attr=vedge_raw.transpose((1, 0, 2)))
        self._state_dict["vo"][tick_range] = np.expand_dims(vo, axis=0)
        self._state_dict["vedge"][tick_range] = np.expand_dims(vedge, axis=0)
        self._state_dict["po"][tick_range] = np.expand_dims(po, axis=0)
        self._state_dict["pedge"][tick_range] = np.expand_dims(pedge, axis=0)
        self._state_dict["ppedge"][tick_range] = self.normalize(full_on_port[-1]).reshape(1, *full_on_port[-1].shape, 1)

    def __call__(self, action_info=None, snapshot_list=None, tick=None):
        if self._only_demo:
            return
        assert((action_info is not None and snapshot_list is not None) or tick is not None)

        if action_info is not None and snapshot_list is not None:
            # Update the state dict.
            static_code = action_info.port_idx
            dynamic_code = action_info.vessel_idx
            if self.last_tick == action_info.tick:
                tick_range = [action_info.tick]
            else:
                tick_range = list(range(self.last_tick + 1, action_info.tick + 1, 1))

            self.last_tick = action_info.tick
            self._sync_raw_features(snapshot_list, tick_range, static_code, dynamic_code)
            tick = action_info.tick

        # State_tick_range is inverse order.
        state_tick_range = np.arange(tick, max(-1, tick - self._tick_buffer), -1)
        v = np.zeros((self._tick_buffer, self.vessel_cnt, self.get_input_dim("v")))
        v[:len(state_tick_range)] = self._state_dict["v"][state_tick_range]
        p = np.zeros((self._tick_buffer, self.port_cnt, self.get_input_dim("p")))
        p[:len(state_tick_range)] = self._state_dict["p"][state_tick_range]

        # True means padding.
        mask = np.ones(self._tick_buffer, dtype=np.bool)
        mask[:len(state_tick_range)] = False
        ret = {
            "tick": state_tick_range,
            "v": v,
            "p": p,
            "vo": self._state_dict["vo"][tick],
            "po": self._state_dict["po"][tick],
            "vedge": self._state_dict["vedge"][tick],
            "pedge": self._state_dict["pedge"][tick],
            "ppedge": self._state_dict["ppedge"][tick],
            "mask": mask,
            "len": len(state_tick_range),
        }

        return ret

    def normalize(self, feature):
        if not self._normalize:
            return feature
        return feature * self._norm_scale

    def get_input_dim(self, agent_code):
        if agent_code in self.port_code_inv_dict or agent_code == "p":
            return len(self.port_features) + (self.node_cnt if self._feature_config.onehot_identity else 0)
        elif agent_code in self.vessel_code_inv_dict or agent_code == "v":
            return len(self.vessel_features) + (self.node_cnt if self._feature_config.onehot_identity else 0)
        elif agent_code == "vedge":
            # v-p edge: (arrival_time, laden to destination)
            return 2
        elif agent_code == "pedge":
            # p-p edge: (laden to destination, )
            return 1
        else:
            raise ValueError("agent not exist!")
