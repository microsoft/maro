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
                Fields are a required piece of the InfluxDB data structure - you cannot have data in InfluxDB without fields. 
                It’s also important to note that fields are not indexed.
                i.e.:{"port1":1024, "port2":2048}
            tag ({Dict}): dictionary of tags, used for query data from database
                Tags are optional. You don’t need to have tags in your data structure, 
                but it’s generally a good idea to make use of them because, unlike fields, tags are indexed. 
                This means that queries on tags are faster and that tags are ideal for storing commonly-queried metadata.
                i.e.:{"ep":5}
            measurement (string): type of field values, used as data table name in database
                The measurement acts as a container for tags, fields, and the time column, 
                and the measurement name is the description of the data that are stored in the associated fields. 
                Measurement names are strings, and, for any SQL users out there, a measurement is conceptually similar to a table.
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
