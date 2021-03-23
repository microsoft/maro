# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import yaml

from maro.cli.utils.params import GlobalPaths


class DetailsReader:
    @staticmethod
    def load_cluster_details(cluster_name: str) -> dict:
        with open(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}/cluster_details.yml", "r") as fr:
            cluster_details = yaml.safe_load(fr)

        return cluster_details

    @staticmethod
    def load_default_user_details(cluster_name: str) -> dict:
        with open(file=f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}/users/default_user", mode="r") as fr:
            user_id = fr.read()

        with open(
            file=f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}/users/{user_id}/user_details",
            mode="r"
        ) as fr:
            user_details = yaml.safe_load(stream=fr)
            return user_details
