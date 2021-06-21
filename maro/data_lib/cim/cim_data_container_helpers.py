# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import urllib.parse
from typing import Union

from yaml import safe_load

from maro.cli.data_pipeline.utils import StaticParameter
from maro.simulator.utils import seed

from .cim_data_container import CimDataContainer
from .cim_data_generator import CimDataGenerator
from .cim_data_loader import load_from_folder
from .cim_real_data_container import CimRealDataContainer
from .cim_real_data_loader import load_real_data_from_folder


class CimDataContainerWrapper:

    def __init__(self, config_path: str, max_tick: int, topology: str):
        self._data_cntr: Union[CimDataContainer, CimRealDataContainer] = None
        self._max_tick = max_tick
        self._config_path = config_path
        self._start_tick = 0

        self._topology = topology
        self._output_folder = os.path.join(
            StaticParameter.data_root, "cim", urllib.parse.quote(self._topology), str(self._max_tick)
        )
        self._meta_path = os.path.join(StaticParameter.data_root, "cim", "meta", "cim.stops.meta.yml")

        self._init_data_container()

    def _init_data_container(self):
        # read config
        with open(self._config_path, "r") as fp:
            conf: dict = safe_load(fp)
        if "input_setting" in conf and conf["input_setting"]["from_files"]:
            if conf["input_setting"]["input_type"] == "real":
                self._data_cntr = real_data_from_files(data_folder=conf["input_setting"]["data_folder"])
            else:
                self._data_cntr = data_from_dumps(dumps_folder=conf["input_setting"]["data_folder"])
        else:
            self._data_cntr = data_from_generator(
                config_path=self._config_path, max_tick=self._max_tick, start_tick=self._start_tick
            )

    def reset(self):
        """Reset data container internal state"""
        self._data_cntr.reset()

    def __getattr__(self, name):
        return getattr(self._data_cntr, name)


def data_from_dumps(dumps_folder: str) -> CimDataContainer:
    """Collect data from dump folder which contains following files:
    ports.csv, vessels.csv, routes.csv, order_proportion.csv, global_order_proportion.txt, misc.yml, stops.bin

    Args:
        dumps_folder(str): Folder contains dumped files.

    Returns:
        CimDataContainer: Data container used to provide cim data related interfaces.
    """
    assert os.path.exists(dumps_folder), f"[CIM Data Container Wrapper] dump folder not exists: {dumps_folder}"

    data_collection = load_from_folder(dumps_folder)
    # set seed to generate data
    seed(data_collection.seed)

    return CimDataContainer(data_collection)


def data_from_generator(config_path: str, max_tick: int, start_tick: int = 0) -> CimDataContainer:
    """Collect data from data generator with configurations.

    Args:
        config_path(str): Path of configuration file (yaml).
        max_tick (int): Max tick to generate data.
        start_tick(int): Start tick to generate data.

    Returns:
        CimDataContainer: Data container used to provide cim data related interfaces.
    """
    edg = CimDataGenerator()

    data_collection = edg.gen_data(config_path, start_tick=start_tick, max_tick=max_tick)

    return CimDataContainer(data_collection)


def real_data_from_files(data_folder: str) -> CimRealDataContainer:
    assert os.path.exists(data_folder), f"[CIM Data Container Wrapper] file folder not exists: {data_folder}"

    data_collection = load_real_data_from_folder(data_folder)
    # set seed to generate data
    seed(data_collection.seed)

    return CimRealDataContainer(data_collection)


__all__ = ['data_from_dumps', 'data_from_generator', 'real_data_from_files']
