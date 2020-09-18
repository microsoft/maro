# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import urllib.parse

from .ecr_data_generator import EcrDataGenerator
from .ecr_data_container import EcrDataContainer
from .ecr_data_loader import load_from_folder
from .ecr_data_dump import dump_from_config

from maro.simulator.utils.sim_random import random, SimRandom
from maro.cli.data_pipeline.utils import convert, StaticParameter


class EcrDataContainerWrapper:

    def __init__(self, config_path: str, max_tick: int, topology: str):
        self._data_cntr: EcrDataContainer = None
        self._max_tick = max_tick
        self._config_path = config_path
        self._start_tick = 0

        self._topology = topology
        self._output_folder = os.path.join(StaticParameter.data_root, "ecr", urllib.parse.quote(self._topology), str(self._max_tick))
        self._meta_path = os.path.join(StaticParameter.data_root, "ecr", "meta", "ecr.stops.meta.yml")

        self._init_data_container()

    def _init_data_container(self):
        self._data_cntr = data_from_generator(config_path=self._config_path, max_tick=self._max_tick, start_tick=self._start_tick)

    def reset(self):
        """Reset data container internal state"""
        self._data_cntr.reset()

    def __getattr__(self, name):
        return getattr(self._data_cntr, name)


def data_from_dumps(dumps_folder: str) -> EcrDataContainer:
    """Collect data from dump folder which contains following files:
    ports.csv, vessels.csv, routes.csv, order_proportion.csv, global_order_proportion.txt, misc.yml, stops.bin


    Args:
        dumps_folder(str): folder contains dumped files

    Returns:
        EcrDataContainer: data container used to provide ecr data related interfaces
    """
    assert os.path.exists(dumps_folder)

    data_collection = load_from_folder(dumps_folder)

    return EcrDataContainer(data_collection)


def data_from_generator(config_path: str, max_tick: int, start_tick: int = 0) -> EcrDataContainer:
    """Collect data from data generator with configurations
    
    
    Args:
        config_path(str): path of configuration file (yaml)
        max_tick (int): max tick to generate data
        start_tick(int): start tick to generate data
    
    Returns:
        EcrDataContainer: data container used to provide ecr data related interfaces
    """
    edg = EcrDataGenerator()

    data_collection = edg.gen_data(config_path, start_tick=start_tick, max_tick=max_tick)

    return EcrDataContainer(data_collection)


__all__ = ['data_from_dumps', 'data_from_generator']
