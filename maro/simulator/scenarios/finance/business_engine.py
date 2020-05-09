import os
import time
import datetime
from typing import Dict, List
from enum import IntEnum
from yaml import safe_load

from maro.simulator.scenarios.entity_base import FrameBuilder
from maro.simulator.event_buffer import EventBuffer, DECISION_EVENT, Event
from maro.simulator.frame import Frame, SnapshotList
from maro.simulator.scenarios.abs_business_engine import AbsBusinessEngine
from maro.simulator.utils.common import total_frames

from .account import Account, AccountSnapshotWrapper
from .common import FinanceType, SubEngineAccessWrapper, TradeResult
from .sub_engines.stock.stock_business_engine import StockBusinessEngine

# type 2 class
sub_engine_definitions = {
    FinanceType.stock: StockBusinessEngine
}

# class FinanceEventType(IntEnum):


class FinanceBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, config_path: str, start_tick: int, max_tick: int, frame_resolution: int):
        super().__init__(event_buffer, config_path, start_tick, max_tick, frame_resolution)

        self._conf = {}
        self._sub_engines = {}
        self._frame_accessor: SubEngineAccessWrapper.PropertyAccessor = None
        self._snapshot_accessor: SubEngineAccessWrapper.PropertyAccessor = None
        self._node_mapping_accessor: SubEngineAccessWrapper.PropertyAccessor = None
        self._sub_engine_accessor: SubEngineAccessWrapper = None
        self._beginning_timestamp = 0

        self._read_conf()
        self._init_sub_engines()

        self._acount = Account(self.snapshots, self._account_frame, self._conf["account"]["money"])  # contain trade result
        
        # out-side accessor as snapshost_list.trade_history
        self._snapshot_accessor.add_item("trade_history", self._acount.trade_history)

        self._register_events()

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
        print("cur tick: ", tick)
        for sub_engine in self._sub_engines.values():
            sub_engine.step(tick)

        return tick + 1 == self._max_tick

    def post_step(self, tick):
        """Post-process at specified tick

        Args:
            tick (int): tick to process
        """
        for sub_engine in self._sub_engines.values():
            sub_engine.post_step(tick)

        self._account_snapshots.insert_snapshot(tick)


    @property
    def configs(self) -> dict:
        """object: Configurations of this business engine"""
        return self._conf

    def rewards(self, actions) -> float:
        """Calculate rewards based on actions

        Args:
            actions(list): Action(s) from agent

        Returns:
            float: reward based on actions
        """
        return [self._acount.calc_reward()]

    def reset(self):
        """Reset business engine"""

        for _, engine in self._sub_engines.items():
            engine.reset()

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

    def _register_events(self):
        self._event_buffer.register_event_handler(DECISION_EVENT, self._on_action_recieved)

    def _on_action_recieved(self, evt: Event):
        actions = evt.payload
        if actions is None:
            return

        for action in actions:
            engine_name = action.sub_engine_name

            if engine_name in self._sub_engines:
                result: TradeResult = self._sub_engines[engine_name].take_action(action, self._acount.remaining_money, evt.tick)
                self._acount.take_trade(result, cur_data = self._sub_engines[engine_name]._stock_list, cur_engine = engine_name)
            else:
                raise "Specified engine not exist."

    def _read_conf(self):
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._conf = safe_load(fp)

            self._beginning_timestamp = datetime.datetime.strptime(self._conf["beginning_date"], "%Y-%m-%d").timestamp()

    def _init_sub_engines(self):
        for sub_conf in self._conf["sub-engines"]:
            engine_type = FinanceType[sub_conf["type"]]

            if engine_type in sub_engine_definitions:
                engine = sub_engine_definitions[engine_type](self._beginning_timestamp, self._start_tick, self._max_tick,
                                                             self._frame_resolution, sub_conf, self._event_buffer)

                self._sub_engines[engine.name] = engine

                self._max_tick = engine.max_tick if self._max_tick <= 0 else max(self._max_tick, engine.max_tick)

        self._account_frame = FrameBuilder.new().add_model(Account, 1).build()
        self._account_snapshots = SnapshotList(self._account_frame, total_frames(self._start_tick, self._max_tick, self._frame_resolution))

        # after we aligned the max tick, then post_init to ask sub-engines to init frame and snapshot
        for _, sub_engine in self._sub_engines.items():
            sub_engine.post_init(self._max_tick)

        self._sub_engine_accessor = SubEngineAccessWrapper(self._sub_engines)
        self._frame_accessor = self._sub_engine_accessor.get_property_access("frame")
        self._snapshot_accessor = self._sub_engine_accessor.get_property_access("snapshot_list")
        self._node_mapping_accessor = self._sub_engine_accessor.get_property_access("name_mapping")

        self._account_snapshot_wrapper = AccountSnapshotWrapper(self._account_snapshots, 
            {name: engine.snapshot_list for name, engine in self._sub_engines.items()})

        # make sure out-side can access account wrapper, as it is not a sub-engine
        self._snapshot_accessor.add_item("account", self._account_snapshot_wrapper)
        
