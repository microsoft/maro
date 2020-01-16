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
    def __init__(self, log_folder, node_name_mapping):
        self._ep = 0       
        self._port_idx2name = node_name_mapping['static']
        self._vessel_idx2name = node_name_mapping['dynamic']

        self._decision_log_dict = dict()
        for port_name in self._port_idx2name.values():
            self._decision_log_dict.setdefault(port_name, {})

        self._log_dumpper_dict = dict()
        for port_name in self._port_idx2name.values():
            self._log_dumpper_dict[port_name] = Logger(tag=f'AAAAA.env_logger.{port_name}', format_=LogFormat.none, dump_folder=log_folder, dump_mode='w', auto_timestamp=False, extension_name='csv')
            self._log_dumpper_dict[port_name].debug("ep,tick,empty,full,on shipper,on consignee,booking,shortage,fulfillment,accumulate booking,accumulate shortage,accumulate fulfillment,vessel name,action,early discharge")
 
    def augment_log_decision_tick(self, decision_event, action):
        tick = decision_event.tick
        port_idx = action.port_idx
        port_name = self._port_idx2name[port_idx]
        vessel_idx = action.vessel_idx
        vessel_name = self._vessel_idx2name[vessel_idx]
        
        self._decision_log_dict[port_name].setdefault(tick, list())
        self._decision_log_dict[port_name][tick].append((vessel_name, action.quantity, decision_event.early_discharge))
  
    def print_env_log(self, max_tick, snapshot_list):
        ports = snapshot_list.static_nodes
        for port_idx, port_name in self._port_idx2name.items():
            for tick in range(max_tick):
                empty = ports[tick: port_idx: ('empty', 0)][0]
                full = ports[tick: port_idx: ('full', 0)][0]
                on_shipper = ports[tick: port_idx: ('on_shipper', 0)][0]
                on_consignee = ports[tick: port_idx: ('on_consignee', 0)][0]
                booking = ports[tick: port_idx: ('booking', 0)][0]
                shortage = ports[tick: port_idx: ('shortage', 0)][0]
                fulfillment = ports[tick: port_idx: ('fulfillment', 0)][0]
                accumulate_booking = ports[tick: port_idx: ('acc_booking', 0)][0]
                accumulate_shortage = ports[tick: port_idx: ('acc_shortage', 0)][0]
                accumulate_fulfillment = ports[tick: port_idx: ('acc_fulfillment', 0)][0]
                self._log_dumpper_dict[port_name].debug(f"{self._ep},{tick},{empty},{full},{on_shipper},{on_consignee},{booking},{shortage},{fulfillment},{accumulate_booking},{accumulate_shortage},{accumulate_fulfillment}")
                
                if port_name in self._decision_log_dict.keys() and tick in self._decision_log_dict[port_name].keys():
                    for vessel_decision in self._decision_log_dict[port_name][tick]:
                        self._log_dumpper_dict[port_name].debug(f",,,,,,,,,,,,{vessel_decision[0]},{vessel_decision[1]},{vessel_decision[2]}")
    
        self._decision_log_dict = dict()
        for port_name in self._port_idx2name.values():
            self._decision_log_dict.setdefault(port_name, {})

        self._ep += 1


