# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import urllib.request
import uuid

import numpy as np

from maro.cli.utils.params import GlobalPaths
from maro.data_lib.binary_converter import BinaryConverter
from maro.utils.exception.cli_exception import CommandError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


def convert(meta: str, file: list, output: str, start_timestamp: int = None, **kwargs):
    meta_file = meta
    csv_files = file
    output_file = output

    if not os.path.exists(meta_file):
        raise CommandError("convert", "meta file not exist.\n")

    if not all([os.path.exists(f) for f in csv_files]):
        raise CommandError("convert", "some source file not exist.\n")

    converter = BinaryConverter(output_file, meta_file, start_timestamp)

    for csv_file in csv_files:
        converter.add_csv(csv_file)


def download_file(source: str, destination: str):
    tmpdir = os.path.join(StaticParameter.data_root, "temp", generate_name_with_uuid())
    temp_file_name = os.path.join(tmpdir, os.path.basename(destination))
    if destination.startswith("~"):
        destination = os.path.expanduser(destination)
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    source_data = urllib.request.urlopen(source)
    res_data = source_data.read()
    with open(temp_file_name, "wb") as f:
        f.write(res_data)

    if os.path.exists(destination):
        os.remove(destination)
    os.rename(temp_file_name, destination)
    os.rmdir(tmpdir)
    logger.info_green("Download finished.")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class StaticParameter:
    data_root = os.path.expanduser(GlobalPaths.MARO_DATA)


def chagne_file_path(source_file_path, target_dir):
    return os.path.join(target_dir, os.path.basename(source_file_path))


def generate_name_with_uuid(uuid_len: int = 16) -> str:
    postfix = uuid.uuid4().hex[:uuid_len]
    return f"{postfix}"
