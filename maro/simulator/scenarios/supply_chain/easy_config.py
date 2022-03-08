# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class EasyConfig(dict):
    """A wrapper base on dictionary, give it ability to access key as property."""

    # Default value for not exist keys.
    default_values: dict = {}

    def __getattr__(self, key):
        if key in self:
            return self[key]

        if key in self.default_values:
            return self.default_values[key]

        return None

    def __setattr__(self, key, value):
        self[key] = value


class SkuInfo(EasyConfig):
    """Sku information wrapper, with default property value."""
    default_values = {
        "price": 0,
        "vlt": 1,
        "product_unit_cost": 0,
        "backlog_ratio": 0,
        "service_level": 0.90,
        "cost": 0,
    }
