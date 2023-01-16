# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import urllib.parse
from typing import Optional

from maro.cli.data_pipeline.utils import StaticParameter
from maro.simulator.utils import random, seed

from .cim_data_container import CimBaseDataContainer, CimRealDataContainer, CimSyntheticDataContainer
from .cim_data_generator import gen_cim_data
from .cim_data_loader import load_from_folder, load_real_data_from_folder
from .utils import DATA_CONTAINER_INIT_SEED_LIMIT, ROUTE_INIT_RAND_KEY


class CimDataContainerWrapper:
    def __init__(self, config_path: str, max_tick: int, topology: str):
        self._data_cntr: Optional[CimBaseDataContainer] = None
        self._max_tick = max_tick
        self._config_path = config_path
        self._start_tick = 0

        self._topology = topology
        self._output_folder = os.path.join(
            StaticParameter.data_root,
            "cim",
            urllib.parse.quote(self._topology),
            str(self._max_tick),
        )
        self._meta_path = os.path.join(StaticParameter.data_root, "cim", "meta", "cim.stops.meta.yml")

        self._init_data_container()

        self._random_seed: Optional[int] = None
        self._re_init_data_cntr_flag: bool = False

    def _init_data_container(self, topology_seed: int = None):
        if not os.path.exists(self._config_path):
            raise FileNotFoundError
        # Synthetic Data Mode: config.yml must exist.
        config_path = os.path.join(self._config_path, "config.yml")
        if os.path.exists(config_path):
            self._data_cntr = data_from_generator(
                config_path=config_path,
                max_tick=self._max_tick,
                start_tick=self._start_tick,
                topology_seed=topology_seed,
            )
        elif os.path.exists(os.path.join(self._config_path, "order_proportion.csv")):
            self._data_cntr = data_from_dumps(dumps_folder=self._config_path)
        else:
            # Real Data Mode: read data from input data files, no need for any config.yml.
            self._data_cntr = data_from_files(data_folder=self._config_path)

    def reset(self, keep_seed: bool) -> None:
        """Reset data container internal state"""
        if not keep_seed:
            self._random_seed = random[ROUTE_INIT_RAND_KEY].randint(0, DATA_CONTAINER_INIT_SEED_LIMIT - 1)
            self._re_init_data_cntr_flag = True

        if self._re_init_data_cntr_flag:
            self._init_data_container(self._random_seed)
            self._re_init_data_cntr_flag = False
        else:
            self._data_cntr.reset()  # Reset the data container with reproduce-ability

    def set_seed(self, random_seed: int) -> None:
        self._random_seed = random_seed
        self._re_init_data_cntr_flag = True

    def __getattr__(self, name):
        return getattr(self._data_cntr, name)


def data_from_dumps(dumps_folder: str) -> CimSyntheticDataContainer:
    """Collect data from dump folder which contains following files:
    ports.csv, vessels.csv, routes.csv, order_proportion.csv, global_order_proportion.txt, misc.yml, stops.bin

    Args:
        dumps_folder(str): Folder contains dumped files.

    Returns:
        CimSyntheticDataContainer: Data container used to provide cim data related interfaces.
    """
    assert os.path.exists(dumps_folder), f"[CIM Data Container Wrapper] dump folder not exists: {dumps_folder}"

    data_collection = load_from_folder(dumps_folder)
    # set seed to generate data
    seed(data_collection.seed)

    return CimSyntheticDataContainer(data_collection)


def data_from_generator(
    config_path: str,
    max_tick: int,
    start_tick: int = 0,
    topology_seed: int = None,
) -> CimSyntheticDataContainer:
    """Collect data from data generator with configurations.

    Args:
        config_path(str): Path of configuration file (yaml).
        max_tick (int): Max tick to generate data.
        start_tick(int): Start tick to generate data.
        topology_seed(int): Random seed of the business engine. \
                'None' means using the seed in the configuration file.

    Returns:
        CimSyntheticDataContainer: Data container used to provide cim data related interfaces.
    """
    data_collection = gen_cim_data(
        config_path,
        start_tick=start_tick,
        max_tick=max_tick,
        topology_seed=topology_seed,
    )

    return CimSyntheticDataContainer(data_collection)


def data_from_files(data_folder: str) -> CimRealDataContainer:
    assert os.path.exists(data_folder), f"[CIM Data Container Wrapper] file folder not exists: {data_folder}"

    data_collection = load_real_data_from_folder(data_folder)
    # set seed to generate data
    seed(data_collection.seed)

    return CimRealDataContainer(data_collection)


__all__ = ["data_from_dumps", "data_from_generator", "data_from_files"]
