from maro.simulator.scenarios.fin.abs_sub_business_engine import AbsSubBusinessEngine
from maro.simulator.frame import SnapshotList, Frame




class FinanceBusinessEngine(AbsSubBusinessEngine):
    def __init__(self, start_tick: int, max_tick: int, frame_resolution: int, config: str, event_buffer: EventBuffer):
        super().__init__(start_tick, max_tick, frame_resolution, config, event_buffer)

    @property
    @abstractmethod
    def snapshot_list(self): 
        pass

    @abstractmethod
    def step(self, tick: int):
        pass

    @abstractmethod
    def post_step(self, tick: int):
        pass