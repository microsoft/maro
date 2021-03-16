# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import namedtuple


ConsumerAction = namedtuple("ConsumerAction", ("id", "source_id", "quantity", "vlt"))

ManufactureAction = namedtuple("ManufactureAction", ("id", "production_rate"))
