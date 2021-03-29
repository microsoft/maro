# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import namedtuple

Order = namedtuple("Order", ("destination", "product_id", "quantity", "vlt"))
