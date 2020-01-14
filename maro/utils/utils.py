# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pickle import loads, dumps
from maro.utils import Logger, LogFormat


def clone(obj):
    """Clone an object"""
    return loads(dumps(obj))


class DottableDict(dict):
    """A wrapper to dictionary to make possible to key as property"""
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


def convert_dottable(natural_dict: dict):
    """Convert a dictionary to DottableDict

    Returns:
        DottableDict: doctable object
    """
    dottable_dict = DottableDict(natural_dict)
    for k, v in natural_dict.items():
        if type(v) is dict:
            v = convert_dottable(v)
            dottable_dict[k] = v
    return dottable_dict

class Env_Logger():
    def __init__(self):
        self._ep = 0

    def init_parameters(self, log_folder, node_name_mapping):
        self._port_idx2name = node_name_mapping['static']
        self._vessel_idx2name = node_name_mapping['dynamic']

        self._log_dict = dict()
        for port_name in self._port_idx2name.values():
            self._log_dict.setdefault(port_name, {})

        self._log_dumpper_dict = dict()
        for port_name in self._port_idx2name.values():
            self._log_dumpper_dict[port_name] = Logger(tag=f'a_env_logger.{port_name}', format_=LogFormat.none, dump_folder=log_folder, dump_mode='w', auto_timestamp=False, extension_name='csv')
            self._log_dumpper_dict[port_name].debug("ep,tick,empty,full,on shipper,on consignee,booking,shortage,fulfillment,accumulate shortage,accumulate shortage,accumulate fulfillment,early discharge,action")

    def augment_log_from_snapshot(self, tick, snapshot_list):
        ports = snapshot_list.static_nodes
        for port_idx, port_name in self._port_idx2name.items():
            self._log_dict[port_name].setdefault(tick, dict())
            self._log_dict[port_name][tick].setdefault("empty", ports[tick: port_idx: ('empty', 0)][0])
            self._log_dict[port_name][tick].setdefault("full", ports[tick: port_idx: ('full', 0)][0])
            self._log_dict[port_name][tick].setdefault("on_shipper", ports[tick: port_idx: ('on_shipper', 0)][0])
            self._log_dict[port_name][tick].setdefault("on_consignee", ports[tick: port_idx: ('on_consignee', 0)][0])
            self._log_dict[port_name][tick].setdefault("booking", ports[tick: port_idx: ('booking', 0)][0])
            self._log_dict[port_name][tick].setdefault("shortage", ports[tick: port_idx: ('shortage', 0)][0])
            self._log_dict[port_name][tick].setdefault("fulfillment", ports[tick: port_idx: ('fulfillment', 0)][0])
            self._log_dict[port_name][tick].setdefault("accumulate_booking", ports[tick: port_idx: ('acc_booking', 0)][0])
            self._log_dict[port_name][tick].setdefault("accumulate_shortage", ports[tick: port_idx: ('acc_shortage', 0)][0])
            self._log_dict[port_name][tick].setdefault("accumulate_fulfillment", ports[tick: port_idx: ('acc_fulfillment', 0)][0])
            self._log_dict[port_name][tick].setdefault("early_discharge", ports[tick: port_idx: ('early_discharge', 0)][0])
            self._log_dict[port_name][tick].setdefault('action', list())

    def augment_log_from_action(self, tick, action):
        port_idx = action.port_idx
        port_name = self._port_idx2name[port_idx]
        vessel_idx = action.vessel_idx
        vessel_name = self._vessel_idx2name[vessel_idx]
        
        self._log_dict[port_name].setdefault(tick, dict())
        self._log_dict[port_name][tick].setdefault('action', list())
        self._log_dict[port_name][tick]['action'].append((vessel_name, action.quantity))
  
    def end_episode_callback(self):
        for port_name in self._port_idx2name.values():
            for tick in self._log_dict[port_name].keys():
                log_for_tick = self._log_dict[port_name][tick]
                self._log_dumpper_dict[port_name].debug(f"{self._ep},{tick},{log_for_tick['empty']},{log_for_tick['full']},{log_for_tick['on_shipper']},{log_for_tick['on_consignee']},{log_for_tick['booking']},{log_for_tick['shortage']},{log_for_tick['fulfillment']},{log_for_tick['accumulate_booking']},{log_for_tick['accumulate_shortage']},{log_for_tick['accumulate_fulfillment']},{log_for_tick['early_discharge']}")
                for vessel_action in log_for_tick['action']:
                    self._log_dumpper_dict[port_name].debug(f",,,,,,,,,,,,,{vessel_action[0]},{vessel_action[1]}")
    
        self._log_dict = dict()
        for port_name in self._port_idx2name.values():
            self._log_dict.setdefault(port_name, {})
        
        self._ep += 1


