# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from enum import Enum


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


class GlobalFilePaths():
    ports_sum = "snapshot_ports_summary.csv"
    vessels_sum = "snapshot_vessels_summary.csv"
    stations_sum = "snapshot_stations_summary.csv"
    name_convert = "name_conversion.csv"


class GlobalScenaios(Enum):
    cim = 1
    citi_bike = 2
