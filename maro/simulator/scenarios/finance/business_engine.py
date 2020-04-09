import os
from typing import Dict, List

from yaml import safe_load

from maro.simulator.event_buffer import EventBuffer
from maro.simulator.frame import Frame, SnapshotList
from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine

from .common import FinanceType
from .sub_engines.stock.stock_business_engine import StockBusinessEngine
from .common import SubEngineAccessWrapper

# type 2 class 
sub_engine_definitions = {
    FinanceType.stock: StockBusinessEngine
}


class FinanceBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, config_path: str, start_tick: int, max_tick: int, frame_resolution: int):
        super().__init__(event_buffer, config_path, start_tick, max_tick, frame_resolution)

        self._conf = {}
        self._sub_engines = []
        self._frame_accessor: SubEngineAccessWrapper.PropertyAccessor = None
        self._snapshot_accessor: SubEngineAccessWrapper.PropertyAccessor = None
        self._node_mapping_accessor: SubEngineAccessWrapper.PropertyAccessor = None
        self._sub_engine_accessor: SubEngineAccessWrapper = None

        self._read_conf()
        self._init_sub_engines()
        
    @property
    def frame(self):
        """Frame: Frame of current business engine
        """
        return self._frame_accessor

    @property
    def snapshots(self):
        """SnapshotList: Snapshot list of current frame"""
        return self._snapshot_accessor

    def step(self, tick: int) -> bool:
        """Used to process events at specified tick, usually this is called by Env at each tick

        Args:
            tick (int): tick to process

        Returns:
            bool: if scenario end at this tick
        """
        for sub_engine in self._sub_engines:
            sub_engine.step(tick)

        return tick + 1 == self._max_tick

    def post_step(self, tick):
        """Post-process at specified tick

        Args:
            tick (int): tick to process
        """
        for sub_engine in self._sub_engines:
            sub_engine.post_step(tick)

    @property
    def configs(self) -> dict:
        """object: Configurations of this business engine"""
        pass

    def rewards(self, actions) -> float:
        """Calculate rewards based on actions

        Args:
            actions(list): Action(s) from agent

        Returns:
            float: reward based on actions
        """
        return []

    def reset(self):
        """Reset business engine"""
        pass

    def get_node_name_mapping(self) -> Dict[str, Dict]:
        """Get node name mappings related with this environment

        Returns:
            Dict[str, Dict]: Node name to index mapping dictionary
        """
        return self._node_mapping_accessor

    def get_agent_idx_list(self) -> List[int]:
        """Get port index list related with this environment

        Returns:
            List[int]: list of port index
        """
        pass

    def _read_conf(self):
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._conf = safe_load(fp)

    def _init_sub_engines(self):
        for sub_conf in self._conf["sub-engines"]:
            engine_type = FinanceType[sub_conf["type"]]

            if engine_type in sub_engine_definitions:
                engine = sub_engine_definitions[engine_type](self._start_tick, self._max_tick, 
                                            self._frame_resolution, sub_conf, self._event_buffer)
                
                self._sub_engines.append(engine)

                self._max_tick = engine.max_tick if self._max_tick <= 0 else min(self._max_tick, engine.max_tick)

        
        # after we aligned the max tick, then post_init to ask sub-engines to init frame and snapshot
        for sub_engine in self._sub_engines:
            sub_engine.post_init(self._max_tick)

        self._sub_engine_accessor = SubEngineAccessWrapper(self._sub_engines)
        self._frame_accessor = self._sub_engine_accessor.get_property_access("frame")
        self._snapshot_accessor = self._sub_engine_accessor.get_property_access("snapshot_list")
        self._node_mapping_accessor = self._sub_engine_accessor.get_property_access("name_mapping")