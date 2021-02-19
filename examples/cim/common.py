# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

PORT_ATTRIBUTES = ["empty", "full", "on_shipper", "on_consignee", "booking", "shortage", "fulfillment"]
VESSEL_ATTRIBUTES = ["empty", "full", "remaining_space"]
ACTION_SPACE = list(np.linspace(-1.0, 1.0, 21))

# Parameters for computing states
LOOK_BACK = 7
MAX_PORTS_DOWNSTREAM = 2

# Parameters for computing rewards
REWARD_TIME_WINDOW = 100
FULFILLMENT_FACTOR = 1.0
SHORTAGE_FACTOR = 1.0
TIME_DECAY = 0.97
