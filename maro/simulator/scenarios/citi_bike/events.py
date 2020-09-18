# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.s


from enum import Enum

class CitiBikeEvents(Enum):
    # Trip related events
    RequireBike = "require_bike"       # customer needs a bike
    ReturnBike = "return_bike"         # customer returns the bike to the target station
    # Rebalance related events
    RebalanceBike = "rebalance_bike"   # rebalance bike event, RL agent (may)need to make a decision
    DeliverBike = "deliver_bike"       # deliver rebalanced bikes to the target station