# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

from .influxdb_proxy import InfluxdbProxy

class Singleton(object):
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance

class DashboardBase(Singleton):
    _connection = None

    @classmethod
    def get_dashboard(cls):
        return cls._instance


    def __init__(self, experiment: str, log_folder: str = None, host: str = 'localhost', port: int = 50301, use_udp: bool = True, udp_port: int = 50304):
        """Setup  dashboard with conf for dashboard

        Args:
            experiment (str): experiment name
            log_folder (str): log folder for logger output

            host (str): influxdb ip address
            port (int): influxdb http port
            use_udp (bool): if use udp port to upload data to influxdb
            udp_port (int): influxdb udp port
        """
        self.experiment = experiment

        if self._connection is None:
            self._connection = InfluxdbProxy(
                host=host, port=port, use_udp=use_udp, udp_port=udp_port, log_folder=log_folder)

    def send(self, fields: dict, tag: dict, measurement: str) -> None:
        """Upload fields to database.

        Args:
            fields ({Dict}): dictionary of field values, key is field name, value is field value. 
                Reference to https://docs.influxdata.com/influxdb/v1.7/concepts/key_concepts/#field-key
                i.e.:{"port1":1024, "port2":2048}
            tag ({Dict}): dictionary of tags, used for query data from database
                Reference to https://docs.influxdata.com/influxdb/v1.7/concepts/key_concepts/#tag-key
                i.e.:{"ep":5}
            measurement (string): type of field values, used as data table name in database
                Reference to https://docs.influxdata.com/influxdb/v1.7/concepts/key_concepts/#measurement
                i.e.:"shortage"
        """
        tag['experiment'] = self.experiment
        self._connection.send(fields=fields, tag=tag, measurement=measurement)

    def upload_to_ranklist(self, ranklist: str, fields: dict) -> None:
        """Upload fields to ranklist table in database.

        Args:
            ranklist (str): a ranklist name
                i.e.: 'test_shortage_ranklist'

            fields ({Dict}): dictionary of field, key is field name, value is field value
                i.e.:{"train":1024, "test":2048}
        """

        measurement = ranklist
        self.send(fields=fields, tag={}, measurement=measurement)
