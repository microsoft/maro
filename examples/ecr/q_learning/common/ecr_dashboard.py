# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from maro.utils.dashboard import DashboardBase


class DashboardECR(DashboardBase):
    def __init__(self, experiment: str, log_folder: str, log_enable: str, host: str = 'localhost', port: int = 50301, use_udp: bool = True, udp_port: int = 50304):
        DashboardBase.__init__(self, experiment, log_folder, log_enable, host, port, use_udp, udp_port)

    def upload_laden_executed(self, nodes_laden_executed, ep):
        """
        Upload scalars to laden_executed table in database.

        Args:
            nodes_laden_executed ({Dict}): dictionary of laden_executed, key is node name, value is node laden_executed value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11

        Returns:
            None.
        """
        nodes_laden_executed['ep'] = ep
        self.send(fields=nodes_laden_executed, tag={
            'experiment': self.experiment}, measurement='laden_executed')

    def upload_laden_planed(self, nodes_laden_planed, ep):
        """
        Upload scalars to laden_planed table in database.

        Args:
            nodes_laden_planed ({Dict}): dictionary of laden_planed, key is node name, value is node laden_planed value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11

        Returns:
            None.
        """
        nodes_laden_planed['ep'] = ep
        self.send(fields=nodes_laden_planed, tag={
            'experiment': self.experiment}, measurement='laden_planed')

    def upload_shortage(self, nodes_shortage, ep):
        """
        Upload scalars to shortage table in database.

        Args:
            nodes_shortage ({Dict}): dictionary of shortage, key is node name, value is node shortage value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11

        Returns:
            None.
        """
        nodes_shortage['ep'] = ep
        self.send(fields=nodes_shortage, tag={
            'experiment': self.experiment}, measurement='shortage')

    def upload_booking(self, nodes_booking, ep):
        """
        Upload scalars to booking table in database.

        Args:
            nodes_booking ({Dict}): dictionary of booking, key is node name, value is node booking value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11

        Returns:
            None.
        """
        nodes_booking['ep'] = ep
        self.send(fields=nodes_booking, tag={
            'experiment': self.experiment}, measurement='booking')

    def upload_q_value(self, nodes_q, ep, action):
        """
        Upload scalars to q_value table in database.

        Args:
            nodes_q ({Dict}): dictionary of q_value, key is node name, value is node q_value value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11
            action (int): current ep of the data, used as scalars information to identify data of different action in database
                i.e.: 0

        Returns:
            None.
        """
        nodes_q['ep'] = ep
        nodes_q['action'] = action
        self.send(fields=nodes_q, tag={
            'experiment': self.experiment}, measurement='q_value')

    def upload_d_error(self, nodes_d_error, ep):
        """
        Upload scalars to d_error table in database.

        Args:
            nodes_d_error ({Dict}): dictionary of d_error, key is node name, value is node d_error value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11

        Returns:
            None.
        """
        nodes_d_error['ep'] = ep
        self.send(fields=nodes_d_error, tag={
            'experiment': self.experiment}, measurement='d_error')

    def upload_loss(self, nodes_loss, ep):
        """
        Upload scalars to loss table in database.

        Args:
            nodes_loss ({Dict}): dictionary of loss, key is node name, value is node loss value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11

        Returns:
            None.
        """
        nodes_loss['ep'] = ep
        self.send(fields=nodes_loss, tag={
            'experiment': self.experiment}, measurement='loss')

    def upload_epsilon(self, nodes_epsilon, ep):
        """
        Upload scalars to epsilon table in database.

        Args:
            nodes_epsilon ({Dict}): dictionary of epsilon, key is node name, value is node epsilon value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11

        Returns:
            None.
        """
        nodes_epsilon['ep'] = ep
        self.send(fields=nodes_epsilon, tag={
            'experiment': self.experiment}, measurement='epsilon')

    def upload_early_discharge(self, nodes_early_discharge, ep):
        """
        Upload scalars to early_discharge table in database.

        Args:
            nodes_laden_planed ({Dict}): dictionary of early_discharge, key is node name, value is node early_discharge value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11

        Returns:
            None.
        """
        nodes_early_discharge['ep'] = ep
        self.send(fields=nodes_early_discharge, tag={
            'experiment': self.experiment}, measurement='early_discharge')


    def upload_delayed_laden(self, nodes_delayed_laden, ep):
        """
        Upload scalars to delayed_laden table in database.

        Args:
            nodes_laden_planed ({Dict}): dictionary of delayed_laden, key is node name, value is node delayed_laden value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11

        Returns:
            None.
        """
        nodes_delayed_laden['ep'] = ep
        self.send(fields=nodes_delayed_laden, tag={
            'experiment': self.experiment}, measurement='delayed_laden')

    def upload_vessel_usage(self, vessel_usage, ep):
        """
        Upload scalars to vessel_usage table in database.

        Args:
            nodes_laden_planed ({Dict}): dictionary of vessel usage, values are vessel name and  vessel usage values
                i.e.:{"vessel":"ship1", "tick":234, "remaining_space":1024, "full":2048, "empty":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11

        Returns:
            None.
        """
        vessel_usage['ep'] = ep
        self.send(fields=vessel_usage, tag={
            'experiment': self.experiment}, measurement='vessel_usage')

    def upload_event_delayed_laden(self, nodes_delayed_laden, ep, tick):
        """
        Upload scalars to event_delayed_laden table in database.

        Args:
            nodes_laden_planed ({Dict}): dictionary of delayed_laden, key is node name, value is node delayed_laden value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11
            tick (int): event tick of the data, used as scalars information to identify data of different tick in database
                i.e.: 132

        Returns:
            None.
        """
        nodes_delayed_laden['ep'] = ep
        nodes_delayed_laden['tick'] = tick
        self.send(fields=nodes_delayed_laden, tag={
            'experiment': self.experiment}, measurement='event_delayed_laden')

    def upload_event_early_discharge(self, nodes_early_discharge, ep, tick):
        """
        Upload scalars to event_early_discharge table in database.

        Args:
            nodes_laden_planed ({Dict}): dictionary of early_discharge, key is node name, value is node early_discharge value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11
            tick (int): event tick of the data, used as scalars information to identify data of different tick in database
                i.e.: 132

        Returns:
            None.
        """
        nodes_early_discharge['ep'] = ep
        nodes_early_discharge['tick'] = tick
        self.send(fields=nodes_early_discharge, tag={
            'experiment': self.experiment}, measurement='event_early_discharge')

    def upload_event_shortage(self, nodes_event_shortage, ep, tick):
        """
        Upload scalars to event_shortage table in database.

        Args:
            nodes_event_shortage ({Dict}): dictionary of event_shortage, key is node name, value is node event_shortage value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11
            tick (int): event tick of the data, used as scalars information to identify data of different tick in database
                i.e.: 132

        Returns:
            None.
        """
        nodes_event_shortage['ep'] = ep
        self.send(fields=nodes_event_shortage, tag={
            'experiment': self.experiment}, measurement='event_shortage')


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
    author = '5000_rl_author'
    commit = '6000_rl_commit'
