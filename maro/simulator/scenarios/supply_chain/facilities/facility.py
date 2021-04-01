# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from abc import ABC
from maro.simulator.scenarios.supply_chain.units.balancesheet import BalanceSheet

class FacilityBase(ABC):
    """Base of all facilities."""

    # Id of this facility.
    id: int = None

    # Name of this facility.
    name: str = None

    # World of this facility belongs to.
    world = None

    # Skus in this facility.
    skus: dict = None

    # Product units for each sku in this facility.
    # Key is sku(product) id, value is the instance of product unit.
    products: dict = None

    # Storage unit in this facility.
    storage = None

    # Distribution unit in this facility.
    distribution = None

    # Upstream facilities.
    # Key is sku id, value is the list of product unit from upstream.
    upstreams: dict = None
    downstreams: dict = None

    # Configuration of this facility.
    configs: dict = None

    data_model_name: str = None
    data_model_index: int = 0

    step_balance_sheet: BalanceSheet = None
    total_balance_sheet: BalanceSheet = None
    children: list = None

    def __init__(self):
        self.upstreams = {}
        self.downstreams = {}

        self.step_balance_sheet = BalanceSheet()
        self.total_balance_sheet = BalanceSheet()
        self.step_reward = 0
        self.children = []

    def parse_skus(self, configs: dict):
        """Parse sku information from config.

        Args:
            configs (dict): Configuration of skus belongs to this facility.
        """
        pass

    def parse_configs(self, configs: dict):
        """Parse configuration of this facility.

        Args:
            configs (dict): Configuration of this facility.
        """
        self.configs = configs

    def get_config(self, key: str, default: object = None):
        return default if self.configs is None else self.configs.get(key, default)

    def initialize(self):
        """Initialize this facility after frame is ready."""
        has_storage = self.storage is not None
        has_distribution = self.distribution is not None

        self.data_model.initialize()

        if self.storage is not None:
            self.children.append(self.storage)

        if self.distribution is not None:
            self.children.append(self.distribution)

        if self.products is not None:
            for product in self.products.values():
                self.children.append(product)

    def step(self, tick: int):
        """Push facility to next step.

        Args:
            tick (int): Current simulator tick.
        """
        rewards = []
        balance_sheets = []

        for unit in self.children:
            unit.step(tick)

            balance_sheets.append(unit.step_balance_sheet)
            rewards.append(unit.step_reward)

        self.step_balance_sheet = sum(balance_sheets)
        self.step_reward = sum(rewards)

        self.total_balance_sheet += self.step_balance_sheet

    def flush_states(self):
        """Flush states into frame."""
        for unit in self.children:
            unit.flush_states()

    def post_step(self, tick: int):
        """Post processing at the end of step."""
        for unit in self.children:
            unit.post_step(tick)

    def reset(self):
        """Reset facility for new episode."""
        for unit in self.children:
            unit.reset()

    def get_in_transit_orders(self):
        in_transit_orders = defaultdict(int)

        for product_id, product in self.products.items():
            if product.consumer is not None:
                in_transit_orders[product_id] = product.consumer.get_in_transit_quantity()

        return in_transit_orders

    def get_node_info(self) -> dict:
        products_info = {}

        for product_id, product in self.products.items():
            products_info[product_id] = product.get_unit_info()

        return {
            "id": self.id,
            "name": self.name,
            "class": type(self),
            "node_index": self.data_model_index,
            "units": {
                "storage": self.storage.get_unit_info() if self.storage is not None else None,
                "distribution": self.distribution.get_unit_info() if self.distribution is not None else None,
                "products": products_info
            },
            "configs": self.configs,
            "skus": self.skus,
            "upstreams": { product_id: [f.id for f in source_list] for product_id, source_list in self.upstreams.items()}
        }
