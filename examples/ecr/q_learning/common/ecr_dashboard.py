# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from maro.utils.dashboard import DashboardBase


class DashboardECR(DashboardBase):
    def __init__(self, experiment: str, log_folder: str, log_enable: str, host: str = 'localhost', port: int = 50301, use_udp: bool = True, udp_port: int = 50304):
        DashboardBase.__init__(self, experiment, log_folder, log_enable, host, port, use_udp, udp_port)

    # for laden_executed, laden_planed, shortage, booking, q_value, d_error, loss, epsilon, early_discharge, delayed_laden
    def upload_ep_data(self, fields, ep, measurement):
        """
        Upload ep data to measurement table in database.

        Args:
            fields ({Dict}): dictionary of ep data, key is ep data name, value is ep data value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as fields information to identify data of different ep in database
                i.e.: 11
            measurement (str): specify the measurement which the data will be stored in.

        Returns:
            None.
        """
        fields['ep'] = ep
        self.send(fields=fields, tag={
            'experiment': self.experiment}, measurement=measurement)

    # for vessel_usage event_delayed_laden, event_early_discharge, event_shortage
    def upload_tick_data(self, fields, ep, tick, measurement):
        """
        Upload tick data to measurement table in database.

        Args:
            fields ({Dict}): dictionary of delayed_laden, key is node name, value is node delayed_laden value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11
            tick (int): event tick of the data, used as scalars information to identify data of different tick in database
                i.e.: 132
            measurement (str): specify the measurement which the data will be stored in.

        Returns:
            None.
        """
        fields['ep'] = ep
        fields['tick'] = tick
        self.send(fields=fields, tag={
            'experiment': self.experiment}, measurement=measurement)

class RanklistColumns(Enum):
    """
    Column names for rank list
    Temporary use X000 to sort columns in rank list dashboard
    TODO: investigate 
    """                        
    experiment = '0000_rl_experiment'
    shortage = '1000_rl_shortage'
    model_size = '2000_rl_model_size'
    train_ep = '3000_rl_train_ep'
    experience_quantity = '4000_rl_experience_quantity'
    initial_lr = '4500_rl_initial_lr'
    author = '5000_rl_author'
    commit = '6000_rl_commit'
