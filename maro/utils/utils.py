# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import configparser
from glob import glob
import io
import numpy as np
import os
from pickle import loads, dumps
import random
import shutil
import time
import warnings

from maro import __data_version__
from maro.utils.exception.cli_exception import CommandError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


def clone(obj):
    """Clone an object"""
    return loads(dumps(obj))


class DottableDict(dict):
    """A wrapper to dictionary to make possible to key as property"""
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


def convert_dottable(natural_dict: dict):
    """Convert a dictionary to DottableDict

    Returns:
        DottableDict: doctable object
    """
    dottable_dict = DottableDict(natural_dict)
    for k, v in natural_dict.items():
        if type(v) is dict:
            v = convert_dottable(v)
            dottable_dict[k] = v
    return dottable_dict


def set_seeds(seed):
    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception as e:
        warnings.warn("Torch not installed.")

    np.random.seed(seed)
    random.seed(seed)


version_file_path = os.path.join(os.path.expanduser("~/.maro"), "version.ini")

project_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

target_source_pairs = [
    (os.path.expanduser("~/.maro/data/citi_bike/meta"),
     os.path.join(project_root, "simulator/scenarios/citi_bike/meta")),
    (os.path.expanduser("~/.maro/data/ecr/meta"),
     os.path.join(project_root, "simulator/scenarios/ecr/meta")),
    (os.path.expanduser("~/.maro/lib/k8s"),
     os.path.join(project_root, "cli/k8s/lib")),
    (os.path.expanduser("~/.maro/lib/grass"),
     os.path.join(project_root, "cli/grass/lib")),
]


def deploy(hide_info=True):
    info_list = []
    error_list = []
    try:
        clean_deployment_folder()
        for target_dir, source_dir in target_source_pairs:
            shutil.copytree(source_dir, target_dir)
        # deploy success
        version_info = configparser.ConfigParser()
        version_info["MARO_DATA"] = {}
        version_info["MARO_DATA"]["version"] = __data_version__
        version_info["MARO_DATA"]["deploy_time"] = str(int(time.time()))
        with io.open(version_file_path, "w") as version_file:
            version_info.write(version_file)
        info_list.append("Data files for MARO deployed.")
    except Exception as e:
        error_list.append(f"An issue occured while deploying meta files for MARO. {e} Please run 'maro meta deploy' to deploy the data files.")

        for target_dir, _ in target_source_pairs:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
    finally:
        if len(error_list) > 0:
            for error in error_list:
                logger.error_red(error)
        elif not hide_info:
            for info in info_list:
                logger.info_green(info)


def check_deployment_status():
    ret = False
    if os.path.exists(version_file_path):
        with io.open(version_file_path, "r") as version_file:
            version_info = configparser.ConfigParser()
            version_info.read(version_file)
            if "MARO_DATA" in version_info \
                and "deploy_time" in version_info["MARO_DATA"] \
                and "version" in version_info["MARO_DATA"] \
                and version_info["MARO_DATA"]["version"] == __data_version__:
                ret = True
    return ret


def clean_deployment_folder():
    for target_dir, _ in target_source_pairs:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
