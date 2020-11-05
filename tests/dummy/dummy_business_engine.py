# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Dummy business engine to test env
"""

from maro.simulator.scenarios import AbsBusinessEngine

from .frame_builder import build_frame


class DummyEngine(AbsBusinessEngine):
    def __init__(self, **kwargs):
        super().__init__(scenario_name="dummy", **kwargs)

        self._frame = build_frame(self.calc_max_snapshots())
        self._dummy_list = self._frame.dummies

    @property
    def frame(self):
        return self._frame

    @property
    def snapshots(self):
        """SnapshotList: Snapshot list of current frame"""
        return self._frame.snapshots

    @property
    def configs(self) -> dict:
        return {"name":"dummy"}

    def step(self, tick: int):
        for dummy_node in self._dummy_list:
            dummy_node.val = tick

    def post_step(self, tick:int):
        if (tick+1) % self._snapshot_resolution == 0:
            self._frame.take_snapshot(self.frame_index(tick))

        # check if we should early stop in step function
        if "post_step_early_stop" in self._additional_options:
            if tick == self._additional_options["post_step_early_stop"]:
                return True

        return tick+1 == self._max_tick

    def reset(self):
        self._frame.reset()
        self._frame.snapshots.reset()

    def get_node_info(self):
        return self._frame.get_node_info()

    def get_agent_idx_list(self):
        return [node.index for node in self._dummy_list]
