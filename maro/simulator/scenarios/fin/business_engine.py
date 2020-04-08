from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine
from maro.simulator.event_buffer import EventBuffer
from yaml import safe_load
from .common import FinanceType
from .sub_engines.stock.stock_business_engine import StockBusinessEngine

# type 2 class 
sub_engine_definitions = {
    FinanceType.Stock: StockBusinessEngine
}


class FinanceBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, config_path: str, start_tick: int, max_tick: int, frame_resolution: int):
        super().__init__(event_buffer, config_path, start_tick, max_tick, frame_resolution)

        self._conf = {}
        self._sub_engines = []

        self._read_conf()
        self._init_sub_engines()

    @property
    def frame(self) -> Frame:
        """Frame: Frame of current business engine
        """
        pass

    @property
    def snapshots(self) -> SnapshotList:
        """SnapshotList: Snapshot list of current frame"""
        pass

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

    @abstractmethod
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
        pass

    def get_agent_idx_list(self) -> List[int]:
        """Get port index list related with this environment

        Returns:
            List[int]: list of port index
        """
        pass

    def _read_conf(self):
        with open(self._config_path) as fp:
            self._conf = safe_load(fp)

    def _init_sub_engines(self):
        for sub_conf in self._conf["sub-engines"]:
            engine_type = sub_conf["type"]

            if engine_type in sub_engine_definitions:
                engine = sub_engine_definitions[engine_type](self._start_tick, self._max_tick, 
                                            self._frame_resolution, sub_conf, self._event_buffer)
                
                self._sub_engines.append(engine)