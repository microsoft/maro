# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl import StateShaper


class CIMStateShaper(StateShaper):
    def __init__(self, *, look_back, max_ports_downstream, port_attributes, vessel_attributes):
        super().__init__()
        self._look_back = look_back
        self._max_ports_downstream = max_ports_downstream
        self._port_attributes = port_attributes
        self._vessel_attributes = vessel_attributes
        self._dim = (look_back + 1) * (max_ports_downstream + 1) * len(port_attributes) + len(vessel_attributes)

    def __call__(self, decision_event, snapshot_list):
        tick, port_idx, vessel_idx = decision_event.tick, decision_event.port_idx, decision_event.vessel_idx
        ticks = [tick - rt for rt in range(self._look_back - 1)]
        future_port_idx_list = snapshot_list["vessels"][tick: vessel_idx: 'future_stop_list'].astype('int')
        port_features = snapshot_list["ports"][ticks: [port_idx] + list(future_port_idx_list): self._port_attributes]
        vessel_features = snapshot_list["vessels"][tick: vessel_idx: self._vessel_attributes]
        state = np.concatenate((port_features, vessel_features))
        return str(port_idx), state

    @property
    def dim(self):
        return self._dim
