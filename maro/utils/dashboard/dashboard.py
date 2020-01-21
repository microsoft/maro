# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

from .influxdb_proxy import InfluxdbProxy as dbProxy

proxy = None
config = None


class DashboardBase():
    _singleton = None
    _connection = None

    def __new__(cls, *a, **k):
        if not cls._singleton:
            cls._singleton = object.__new__(cls)
        return cls._singleton

    def __init__(self, experiment: str, log_folder: str, log_enable: str, host: str = 'localhost', port: int = 50301, use_udp: bool = True, udp_port: int = 50304):
        """Setup  dashboard with conf for dashboard

        Args:
            experiment (str): experiment name
            log_folder (str): log folder for logger output

            host (str): influxdb ip address
            port (int): influxdb http port
            use_udp (bool): if use udp port to upload data to influxdb
            udp_port (int): influxdb udp port
        """
        self.config = None
        self.experiment = experiment

        if self._connection is None:
            self._connection = dbProxy(
                host=host, port=port, use_udp=use_udp, udp_port=udp_port, log_folder=log_folder, log_enable=log_enable)

    def send(self, fields: dict, tag: dict, measurement: str) -> None:
        """Upload fields to database.

        Args:
            fields ({Dict}): dictionary of field values, key is field name, value is field value
                i.e.:{"port1":1024, "port2":2048}
            tag ({Dict}): dictionary of tags, used for query data from database
                i.e.:{"ep":5}
            measurement (string): type of field values, used as data table name in database
                i.e.:"shortage"
        """
        tag['experiment'] = self.experiment
        self._connection.send(fields=fields, tag=tag, measurement=measurement)

    def upload_to_ranklist(self, ranklist: str, fields: dict) -> None:
        """Upload fields to ranklist table in database.

        Args:
            ranklist ({str}): a ranklist name
                i.e.: 'test_shortage_ranklist'

            fields ({Dict}): dictionary of field, key is field name, value is field value
                i.e.:{"train":1024, "test":2048}
        """

        measurement = ranklist
        self.send(fields=fields, tag={}, measurement=measurement)
