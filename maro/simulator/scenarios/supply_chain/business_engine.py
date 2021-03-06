
import os

from maro.simulator.scenarios import AbsBusinessEngine

from maro.event_buffer import MaroEvents, CascadeEvent, AtomEvent

from .units import UnitBase
from .world import World
from .configs import test_world_config


class SupplyChainBusinessEngine(AbsBusinessEngine):
    def __init__(self, **kwargs):
        super().__init__(scenario_name="supply_chain", **kwargs)

        self._register_events()

        self._build_world()

        self._node_mapping = self.world.get_node_mapping()

        self._frame = self.world.frame

        self._action_steps = self.world.configs["action_steps"]

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
        return self._node_mapping

    def step(self, tick: int):
        for _, facility in self.world.facilities.items():
            facility.step(tick)

        # TODO: debug only, ask for action per 5 ticks.
        if tick % self._action_steps == 0:
            decision_event = self._event_buffer.gen_decision_event(tick, None)

            self._event_buffer.insert_event(decision_event)

    def post_step(self, tick: int):
        # take snapshot
        if (tick + 1) % self._snapshot_resolution == 0:
            self._frame.take_snapshot(self.frame_index(tick))

            # TODO: anything need to reset per tick?

        for facility in self.world.facilities.values():
            facility.post_step(tick)

        return tick+1 == self._max_tick

    def reset(self):
        self._frame.reset()
        self._frame.snapshots.reset()

        # TODO: reset frame nodes.
        for _, facility in self.world.facilities.items():
            facility.reset()

    def _register_events(self):
        self._event_buffer.register_event_handler(MaroEvents.TAKE_ACTION, self._on_action_received)

    def _build_world(self):
        self.update_config_root_path(__file__)

        # config_path = os.path.join(self._config_path, "config.yml")

        self.world = World()

        self.world.build(test_world_config, self.calc_max_snapshots())

    def _on_action_received(self, event):
        action = event.payload

        if action:
            # NOTE:
            # we assume that the action is a dictionary that
            # key is the id of unit
            # value is the action for specified unit, the type may different by different type

            for unit_id, control_action in action.items():
                # try to find the unit
                unit: UnitBase = self.world.get_entity(unit_id)

                # dispatch the action
                unit.set_action(control_action)
