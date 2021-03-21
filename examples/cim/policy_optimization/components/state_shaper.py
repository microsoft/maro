# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl import StateShaper

PORT_ATTRIBUTES = ["empty", "full", "on_shipper", "on_consignee", "booking", "shortage", "fulfillment"]
VESSEL_ATTRIBUTES = ["empty", "full", "remaining_space"]


class CIMStateShaper(StateShaper):
    def __init__(self, *, look_back, max_ports_downstream):
        super().__init__()
        self._look_back = look_back
        self._max_ports_downstream = max_ports_downstream
        self._dim = (look_back + 1) * (max_ports_downstream + 1) * len(PORT_ATTRIBUTES) + len(VESSEL_ATTRIBUTES)

    def __call__(self, decision_event, snapshot_list):
        tick, port_idx, vessel_idx = decision_event.tick, decision_event.port_idx, decision_event.vessel_idx
        ticks = [tick - rt for rt in range(self._look_back - 1)]
        future_port_idx_list = snapshot_list["vessels"][tick: vessel_idx: 'future_stop_list'].astype('int')
        port_features = snapshot_list["ports"][ticks: [port_idx] + list(future_port_idx_list): PORT_ATTRIBUTES]
        vessel_features = snapshot_list["vessels"][tick: vessel_idx: VESSEL_ATTRIBUTES]
        state = np.concatenate((port_features, vessel_features))
        return str(port_idx), state

    @property
    def dim(self):
        return self._dim
