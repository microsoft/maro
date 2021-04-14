# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import namedtuple

ConsumerAction = namedtuple("ConsumerAction", ("id", "product_id", "source_id", "quantity", "vlt", "reward_discount"))

ManufactureAction = namedtuple("ManufactureAction", ("id", "production_rate"))
