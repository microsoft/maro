
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
        pass

    def step(self, tick: int):
        for _, facility in self.world.facilities.items():
            facility.step(tick)

    def post_step(self, tick: int):

        return tick+1 == self._max_tick

    def reset(self):
        self._frame.reset()
        self._frame.snapshots.reset()

        # TODO: reset frame nodes.

    def _register_events(self):
        self._event_buffer.register_event_handler(MaroEvents.TAKE_ACTION, self._on_action_recieved)

    def _build_world(self):
        self.update_config_root_path(__file__)

        config_path = os.path.join(self._config_path, "config.yml")

        self.world = World()

        self.world.build(test_world_config)


    def _on_action_recieved(self, event):
        action = event.payload

        if action:
            pass

            # TODO: how to dispatch it to units?
