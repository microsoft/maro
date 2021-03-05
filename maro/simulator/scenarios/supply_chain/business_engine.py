
import os

from maro.simulator.scenarios import AbsBusinessEngine

from maro.event_buffer import MaroEvents, CascadeEvent, AtomEvent

from .world import World
from .configs import test_world_config


class SupplyChainBusinessEngine(AbsBusinessEngine):
    def __init__(self, **kwargs):
        super().__init__(scenario_name="supply_chain", **kwargs)

        self._register_events()

        self._build_world()

        self._frame = self.world.frame

    @property
    def frame(self):
        return self._frame

    @property
    def snapshots(self):
        return self._frame.snapshots

    @property
    def configs(self):
        return self.world.configs

    def get_node_mapping(self) -> dict:
        return self.world.unit_id2index_mapping

    def step(self, tick: int):
        for _, facility in self.world.facilities.items():
            facility.step(tick)

    def post_step(self, tick: int):
        # take snapshot
        if (tick + 1) % self._snapshot_resolution == 0:
            self._frame.take_snapshot(self.frame_index(tick))

            # TODO: anything need to reset per tick?

        return tick+1 == self._max_tick

    def reset(self):
        self._frame.reset()
        self._frame.snapshots.reset()

        # TODO: reset frame nodes.
        for _, facility in self.world.facilities.items():
            facility.reset()

    def _register_events(self):
        self._event_buffer.register_event_handler(MaroEvents.TAKE_ACTION, self._on_action_recieved)

    def _build_world(self):
        self.update_config_root_path(__file__)

        config_path = os.path.join(self._config_path, "config.yml")

        self.world = World()

        self.world.build(test_world_config, self.calc_max_snapshots())

    def _on_action_recieved(self, event):
        action = event.payload

        if action:
            pass

            # TODO: how to dispatch it to units?
