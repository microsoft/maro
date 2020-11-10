# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import logging


class GlobalParams:
    PARALLELS = 5
    LOG_LEVEL = logging.INFO


class GlobalPaths:
    MARO_LIB = '~/.maro/lib'
    MARO_GRASS_LIB = '~/.maro/lib/grass'
    MARO_K8S_LIB = '~/.maro/lib/k8s'
    MARO_CLUSTERS = '~/.maro/clusters'
    MARO_DATA = '~/.maro/data'
    MARO_TEST = '~/.maro/test'
    MARO_INSPECTOR_FILE_PATH = {'ports_file_path': r'snapshot_ports_summary.csv',
                                'vessels_file_path': r'snapshot_vessels_summary.csv',
                                'stations_file_path': r'snapshot_stations_summary.csv',
                                'name_conversion_path': r'name_conversion.csv'}
