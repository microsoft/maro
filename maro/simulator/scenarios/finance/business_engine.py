import os
import time
import datetime
import pandas as pd
from typing import Dict, List
from enum import IntEnum
from yaml import safe_load
from collections import OrderedDict

from maro.data_lib import BinaryReader
from maro.event_buffer import DECISION_EVENT, Event, EventBuffer
from maro.simulator.scenarios import AbsBusinessEngine
from maro.utils.logger import CliLogger
from maro.backends.frame import FrameBase, SnapshotList



from .frame_builder import build_frame
from maro.simulator.utils.common import total_frames

from .account import Account, AccountSnapshotWrapper
from .common import FinanceType, SubEngineAccessWrapper, TradeResult, ActionType, ActionState
from .sub_engines.stock.stock_business_engine import StockBusinessEngine
from .currency import Exchanger

# type 2 class
sub_engine_definitions = {
    FinanceType.stock: StockBusinessEngine
}

logger = CliLogger(name=__name__)


class FinanceBusinessEngine(AbsBusinessEngine):
    def __init__(self, event_buffer: EventBuffer, topology: str, start_tick: int, max_tick: int, snapshot_resolution: int, max_snapshots: int, additional_options: dict = {}):
        super().__init__(
            "finance", event_buffer, topology, start_tick, max_tick,
            snapshot_resolution, max_snapshots, additional_options
            )

        self._conf = {}
        self.update_config_root_path(__file__)
        self._sub_engines = {}
        self._frame_accessor: SubEngineAccessWrapper.PropertyAccessor = None
        self._snapshot_accessor: SubEngineAccessWrapper.PropertyAccessor = None
        self._node_mapping_accessor: SubEngineAccessWrapper.PropertyAccessor = None
        self._sub_engine_accessor: SubEngineAccessWrapper = None
        self._beginning_timestamp = 0
        self._finished_action = OrderedDict()

        self._read_conf()
        self._init_sub_engines()
        self._exchanger = Exchanger(self._conf['account']['exchange_path'], pd.to_datetime(self._conf['beginning_date']))

        

        # out-side accessor as snapshost_list.trade_history
        self._snapshot_accessor.add_item("action_history", self._account.action_history)
        self._snapshot_accessor.add_item("finished_action", self._finished_action)

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
        #print("cur tick: ", tick)
        for sub_engine in self._sub_engines.values():
            sub_engine.step(tick)

        

    def post_step(self, tick):
        """Post-process at specified tick

        Args:
            tick (int): tick to process
        """
        for sub_engine in self._sub_engines.values():
            sub_engine.post_step(tick)

        if (tick + 1) % self._snapshot_resolution == 0:
            # NOTE: we should use frame_index method to get correct index in snapshot list
            self._frame_accessor.take_snapshot(self.frame_index(tick))

        return tick + 1 == self._max_tick

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
        return [self._account.calc_reward()]

    def reset(self):
        """Reset business engine"""

        for _, engine in self._sub_engines.items():
            engine.reset()

        self._account.reset()
        self._finished_action.clear()

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
            if action is None:
                pass
            else:
                if action.id not in self._account.action_history:
                    self._account.action_history[action.id] = action
                engine_name = action.sub_engine_name

                if engine_name in self._sub_engines:
                    if action.action_type == ActionType.transfer:
                        self._account.take_action(action, evt.tick)
                    elif action.action_type == ActionType.cancel_order:
                        self._sub_engines[engine_name].cancel_order(action)
                    elif action.action_type == ActionType.order:
                        result: TradeResult = self._sub_engines[engine_name].take_action(action, self._account._sub_account_dict[engine_name].remaining_money, evt.tick)
                        self._account.take_trade(result, cur_data=self._sub_engines[engine_name]._stock_list, cur_engine=engine_name)
                else:
                    raise "Specified engine not exist."
                if action.state != ActionState.pending and action.id not in self._finished_action:
                    self._finished_action[action.id] = action


    def _read_conf(self):
        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._conf = safe_load(fp)

            self._beginning_timestamp = datetime.datetime.strptime(self._conf["beginning_date"], "%Y-%m-%d").timestamp()

    def _init_sub_engines(self):
        for sub_conf in self._conf["sub-engines"]:
            engine_type = FinanceType[sub_conf["type"]]

            if engine_type in sub_engine_definitions:
                engine = sub_engine_definitions[engine_type](self._beginning_timestamp, self._start_tick, self._max_tick,
                                                             self._snapshot_resolution, sub_conf, self._event_buffer)

                self._sub_engines[engine.name] = engine

                self._max_tick = engine.max_tick if self._max_tick <= 0 else max(self._max_tick, engine.max_tick)

        self._account_frame = build_frame(total_frames(self._start_tick, self._max_tick, self._snapshot_resolution))
        self._account_snapshots = self._account_frame.snapshots
        self._account = self._account_frame.account[0]
        self._account.set_init_state({name: sub_engine.account for name, sub_engine in self._sub_engines.items()}, self._conf['account']['money'])

        self._sub_engine_accessor = SubEngineAccessWrapper(self._sub_engines)
        self._frame_accessor = self._sub_engine_accessor.get_property_access("frame")
        self._snapshot_accessor = self._sub_engine_accessor.get_property_access("snapshot_list")
        self._node_mapping_accessor = self._sub_engine_accessor.get_property_access("name_mapping")

        self._account_snapshot_wrapper = AccountSnapshotWrapper(self._account_snapshots, {name: engine.snapshot_list for name, engine in self._sub_engines.items()})

        # make sure out-side can access account wrapper, as it is not a sub-engine
        self._snapshot_accessor.add_item("account", self._account_snapshot_wrapper)
        self._frame_accessor.add_item("get_node_info", self._account_frame.get_node_info)
        self._frame_accessor.add_item("take_snapshot", self._account_frame.take_snapshot)
