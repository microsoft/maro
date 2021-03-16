# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .unitbase import UnitBase


class SkuUnit(UnitBase):
    """A sku related unit."""

    # Product id (sku id), 0 means invalid.
    product_id: int = 0
