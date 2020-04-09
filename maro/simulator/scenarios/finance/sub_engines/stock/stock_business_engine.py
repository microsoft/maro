from maro.simulator.scenarios.finance.abs_sub_business_engine import AbsSubBusinessEngine
from maro.simulator.frame import SnapshotList, Frame
from maro.simulator.scenarios.finance.common import FinanceType
from maro.simulator.event_buffer import EventBuffer, Event



class StockBusinessEngine(AbsSubBusinessEngine):
    def __init__(self, start_tick: int, max_tick: int, frame_resolution: int, config: dict, event_buffer: EventBuffer):
        super().__init__(start_tick, max_tick, frame_resolution, config, event_buffer)

    @property
    def finance_type(self):
        return FinanceType.Stock

    @property
    def snapshot_list(self): 
        pass

    def step(self, tick: int):
        pass

    def post_step(self, tick: int):
        pass