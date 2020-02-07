# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import datetime
import time
import os

from influxdb import InfluxDBClient
from maro.utils import Logger, LogFormat


class InfluxdbProxy:
    """Provide data upload function for dashboard
    """

    def __init__(self, host: str = 'localhost', port: int = 50301,
                 dbname: str = 'maro', timeout: int = 60,
                 use_udp: bool = True, udp_port: int = 50304,
                 log_folder: str = None):
                 
        self.client = InfluxDBClient(
            host, port, None, None, dbname, timeout=timeout, use_udp=use_udp, udp_port=udp_port)
        self._log_enable = False if log_folder is None else True
        if self._log_enable:
            self.logger = Logger(tag='dashboard', format_=LogFormat.simple,
                                dump_mode='w', auto_timestamp=False,
                                dump_folder=log_folder)

        self.start_time = time.time()

        try:
            self.client.create_database(dbname)
        except Exception as e:
            if self._log_enable:
                self.logger.warn(
                    f"failed to create db for dashboard, please check your network or parameter. {str(e)}")

    def send(self, fields, tag,  measurement):
        """Send fields data to influxdb.

        Args:
            fields ({Dict}): dictionary of scalar, key is scalar name, value is scalar value
                i.e.:{"port1":1024, "port2":2048}
            tag ({Dict}): dictionary of information, used for query data from database
                i.e.:{"ep":5, "experiment":"test_dashboard_01"}
            measurement (string): type of scalars, used as data table name in database
                i.e.:"shortage"
        """
        msg = [{
            'measurement': measurement,
            'fields': fields,
            'time': datetime.datetime.utcnow(),
            'tags': tag
        }]

        try:
            self.client.write_points(msg)
        except Exception as e:
            if self._log_enable:
                self.logger.warn(
                    f'Failed to upload msg via Influxdb, please check your network or parameter . '
                    f'{str(e)} {fields} {tag} {measurement}')
