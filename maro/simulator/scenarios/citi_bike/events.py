# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum


class CitiBikeEvents(Enum):
    """Trip related events."""
    # customer needs a bike
    RequireBike = "require_bike"
    # customer returns the bike to the target station
    ReturnBike = "return_bike"
    # Rebalance related events
    # rebalance bike event, RL agent (may)need to make a decision
    RebalanceBike = "rebalance_bike"
    # deliver rebalanced bikes to the target station
    DeliverBike = "deliver_bike"
