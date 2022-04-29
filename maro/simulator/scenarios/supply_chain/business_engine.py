# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import List

from maro.backends.frame import FrameBase
from maro.event_buffer import CascadeEvent, MaroEvents
from maro.simulator.scenarios import AbsBusinessEngine

from .actions import SupplyChainAction
from .objects import SupplyChainEntity
from .parser import ConfigParser, SupplyChainConfiguration
from .units import ProductUnit
from .world import World


class SupplyChainBusinessEngine(AbsBusinessEngine):
    def __init__(self, **kwargs) -> None:
        super().__init__(scenario_name="supply_chain", **kwargs)

        self._register_events()

        self._build_world()

        self._product_units: List[ProductUnit] = []

        # Prepare product unit for later using.
        for unit in self.world.units.values():
            if isinstance(unit, ProductUnit):
                self._product_units.append(unit)

        self._frame = self.world.frame

        self._node_mapping = self.world.get_node_mapping()

        self._metrics_cache = None

    @property
    def frame(self) -> FrameBase:
        return self._frame

    @property
    def snapshots(self) -> object:
        return self._frame.snapshots

    @property
    def configs(self) -> SupplyChainConfiguration:
        return self.world.configs

    def step(self, tick: int) -> None:
        # Clear the metrics cache.
        self._metrics_cache = None

        """
        Initialize info & status that would be used in step(), including:
        - update SKU price
        - initialize internal status
        """
        for facility in self.world.facilities.values():
            facility.pre_step(tick)

        # TODO: need to order Facility or not?
        # Call step functions by facility
        for facility in self.world.facilities.values():
            facility.step(tick)

        """
        Flush states to frame before generating decision event.
        . The processing logic requires that: DO NOT call flush_states() immediately after step().
        E.g. the ProductUnit.flush_states() should be called after the DistributionUnit.step().
        """
        for facility in self.world.facilities.values():
            facility.flush_states()

        # We do not have payload here.
        decision_event = self._event_buffer.gen_decision_event(tick, None)

        self._event_buffer.insert_event(decision_event)

    def post_step(self, tick: int) -> bool:
        # Call post_step functions by facility.
        for facility in self.world.facilities.values():
            facility.post_step(tick)

        for facility in self.world.facilities.values():
            facility.flush_states()

        self._frame.take_snapshot(self.frame_index(tick))

        return tick + 1 == self._max_tick

    def reset(self, keep_seed: bool = False) -> None:
        self._frame.reset()

        if self._frame.snapshots:
            self._frame.snapshots.reset()

        # Call reset functions by facility.
        for facility in self.world.facilities.values():
            facility.reset()

    def get_node_mapping(self) -> dict:
        return self._node_mapping

    def get_entity_list(self) -> List[SupplyChainEntity]:
        """Get a list of entities.

        Returns:
            list: List of entities.
        """
        return self.world.entity_list

    def _register_events(self) -> None:
        self._event_buffer.register_event_handler(MaroEvents.TAKE_ACTION, self._on_action_received)

    def _build_world(self) -> None:
        self.update_config_root_path(__file__)

        # Core configuration always in topologies folder.
        be_root = os.path.split(os.path.realpath(__file__))[0]
        core_config = os.path.join(be_root, "topologies", "core.yml")

        parser = ConfigParser(core_config, self._config_path)

        conf = parser.parse()

        self.world = World()

        self.world.build(conf, self.calc_max_snapshots(), self._max_tick)

    def _on_action_received(self, event: CascadeEvent) -> None:
        assert isinstance(event.payload, list)
        actions = event.payload
        for action in actions:
            assert isinstance(action, SupplyChainAction)
            entity = self.world.get_entity_by_id(action.id)
            entity.on_action_received(event.tick, action)

    def get_metrics(self) -> dict:
        if self._metrics_cache is None:
            self._metrics_cache = {
                "products": {
                    product.id: {
                        "sale_mean": product.get_sale_mean(),
                        "sale_std": product.get_sale_std(),
                        "selling_price": product.get_max_sale_price(),
                        "pending_order_daily":
                            None if product.consumer is None else product.consumer.pending_order_daily,
                    } for product in self._product_units
                },
                "facilities": {
                    facility.id: {
                        "in_transit_orders": facility.get_in_transit_orders(),
                        "pending_order":
                            None if facility.distribution is None
                            else facility.distribution.get_pending_product_quantities(),
                    } for facility in self.world.facilities.values()
                }
            }

        return self._metrics_cache
