# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

common_config = {
    "port_attributes": ["empty", "full", "on_shipper", "on_consignee", "booking", "shortage", "fulfillment"],
    "vessel_attributes": ["empty", "full", "remaining_space"],
    "action_space": list(np.linspace(-1.0, 1.0, 21)),
    # Parameters for computing states
    "look_back": 7,
    "max_ports_downstream": 2,
    # Parameters for computing actions
    "finite_vessel_space": True,
    "has_early_discharge": True,
    # Parameters for computing rewards
    "reward_eval_delay": 99,
    "fulfillment_factor": 1.0,
    "shortage_factor": 1.0,
    "time_decay": 0.97
}
