# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import psutil
from enum import Enum
from maro.utils.dashboard import DashboardBase


class DashboardECR(DashboardBase):
    # info dictionary
    static_info = {}
    dynamic_info = {}
    ranklist_info = {}


    def __init__(self, experiment: str, log_folder: str, host: str = 'localhost', port: int = 50301, use_udp: bool = True, udp_port: int = 50304):
        DashboardBase.__init__(self, experiment, log_folder, host, port, use_udp, udp_port)
        DashboardECR.static_info['name'] = experiment
        psutil.cpu_percent()

    # for laden_executed, laden_planed, shortage, booking, q_value, d_error, loss, epsilon, early_discharge, delayed_laden
    # for vessel_usage event_delayed_laden, event_early_discharge, event_shortage
    def upload_exp_data(self, fields: dict, ep: int, tick: int, measurement: str) -> None:
        """
        Upload tick data to measurement table in database.

        Args:
            fields ({Dict}): dictionary of the experiment data, key is experiment data name, value is experiment data value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as experiment data information to identify data of different ep in database
                i.e.: 11
            tick (int): event tick of the data, used as experiment data information to identify data of different tick in database
                i.e.: 132
            measurement (str): specify the measurement which the data will be stored in.

        Returns:
            None.
        """
        tag = {}
        if 'COMPONENT_NAME' in DashboardECR.static_info:
            tag['component_name'] = DashboardECR.static_info['COMPONENT_NAME']
        fields['ep'] = ep
        if tick is not None:
            fields['tick'] = tick
        self.send(fields=fields, tag=tag, measurement=measurement)
    
    def update_static_info(self, info:dict) -> None:
        for info_name, info_value in info.items():
            DashboardECR.static_info[info_name] = info_value
        
    def update_dynamic_info(self, info:dict) -> None:
        for info_name, info_value in info.items():
            DashboardECR.dynamic_info[info_name] = info_value
        DashboardECR.dynamic_info['cpu_percent'] = psutil.cpu_percent()
        memory_usage = dict(psutil.virtual_memory()._asdict())
        for memory_usage_name, memory_usage_value in memory_usage.items():
            DashboardECR.dynamic_info[f'memory_{memory_usage_name}'] = memory_usage_value

    def update_ranklist_info(self, info:dict) -> None:
        for info_name, info_value in info.items():
            DashboardECR.ranklist_info[info_name] = info_value

    def upload_to_ranklist(self, ranklist: str, fields: dict) -> None:
        for info_name, info_value in DashboardECR.ranklist_info.items():
            fields[info_name] = info_value
        super(DashboardECR, self).upload_to_ranklist(ranklist=ranklist, fields=fields)
        



class RanklistColumns(Enum):
    """
    Column names for rank list
    Temporary use X000 to sort columns in rank list dashboard
    TODO: investigate better way of sorting the rank list columns
    """                        
    experiment = '0000_rl_experiment'
    shortage = '1000_rl_shortage'
    model_size = '2000_rl_model_size'
    train_ep = '3000_rl_train_ep'
    experience_quantity = '4000_rl_experience_quantity'
    initial_lr = '4500_rl_initial_lr'
    author = '5000_rl_author'
    commit = '6000_rl_commit'
