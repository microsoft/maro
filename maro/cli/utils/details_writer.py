# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

import yaml

from maro.cli.utils.params import GlobalPaths


class DetailsWriter:
    @staticmethod
    def save_cluster_details(cluster_name: str, cluster_details: dict) -> None:
        os.makedirs(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}", exist_ok=True)
        with open(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}/cluster_details.yml", "w") as fw:
            yaml.safe_dump(cluster_details, fw)
