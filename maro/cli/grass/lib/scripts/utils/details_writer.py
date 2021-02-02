# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

import yaml

from .params import Paths


class DetailsWriter:
    """Writer class for details.
    """

    @staticmethod
    def save_local_cluster_details(cluster_details: dict) -> None:
        os.makedirs(name=f"{Paths.ABS_MARO_LOCAL}/cluster", exist_ok=True)
        with open(file=f"{Paths.ABS_MARO_LOCAL}/cluster/cluster_details.yml", mode="w") as fw:
            yaml.safe_dump(data=cluster_details, stream=fw)

    @staticmethod
    def save_local_master_details(master_details: dict) -> None:
        os.makedirs(name=f"{Paths.ABS_MARO_LOCAL}/cluster", exist_ok=True)
        with open(file=f"{Paths.ABS_MARO_LOCAL}/cluster/master_details.yml", mode="w") as fw:
            yaml.safe_dump(data=master_details, stream=fw)
