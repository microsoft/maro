# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


# TODO: original code included the order in raw state, but calculate the price in final state
# so we do not need to put it in the frame, just calculate the total_price per tick/step.

from collections import namedtuple

Order = namedtuple("Order", ("destination", "product_id", "quantity", "vlt"))
