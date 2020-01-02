# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.utils.dashboard import DashboardBase


class DashboardECR(DashboardBase):
    def __init__(self, experiment: str, log_folder: str):
        DashboardBase.__init__(self, experiment, log_folder)

    def upload_laden_executed(self, nodes_laden_executed, ep):
        """
        Upload scalars to laden_executed table in database.

        Args:
            nodes_laden_executed ({Dict}): dictionary of d_error, key is node name, value is node laden_executed value
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
            nodes_laden_planed ({Dict}): dictionary of d_error, key is node name, value is node laden_planed value
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
            nodes_shortage ({Dict}): dictionary of d_error, key is node name, value is node shortage value
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
            nodes_booking ({Dict}): dictionary of d_error, key is node name, value is node booking value
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
            nodes_q ({Dict}): dictionary of d_error, key is node name, value is node q_value value
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
            nodes_loss ({Dict}): dictionary of d_error, key is node name, value is node loss value
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
            nodes_epsilon ({Dict}): dictionary of d_error, key is node name, value is node epsilon value
                i.e.:{"port1":1024, "port2":2048}
            ep (int): current ep of the data, used as scalars information to identify data of different ep in database
                i.e.: 11

        Returns:
            None.
        """
        nodes_epsilon['ep'] = ep
        self.send(fields=nodes_epsilon, tag={
            'experiment': self.experiment}, measurement='epsilon')
