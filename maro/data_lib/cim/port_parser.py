# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List

from .entities import NoisedItem, SyntheticPortSetting


class PortsParser:
    """Parser used to parse port information from configurations
    """

    def parse(self, conf: dict, total_container: int) -> (Dict[str, int], List[SyntheticPortSetting]):
        """Parse specified port configurations.

        Args:
            conf (dict): Configuration to parse.
            total_container (int): Total container in current environment, used to calculate initial empty.

        Returns:
            (Dict[str, int], List[SyntheticPortSetting]): Port mapping (name to index), list of port settings.
        """
        # sum of ratio cannot large than 1
        total_ratio = sum([p["initial_container_proportion"] for p in conf.values()])

        assert round(total_ratio, 7) == 1

        ports_mapping: Dict[str, int] = {}

        index = 0

        # step 1: create mapping
        for port_name, port_info in conf.items():
            ports_mapping[port_name] = index
            index += 1

        port_settings: List[SyntheticPortSetting] = []

        for port_idx, port in enumerate(conf.items()):
            port_name, port_info = port

            # step 2: update initial values
            empty_ratio = port_info["initial_container_proportion"]

            # full return buffer configurations
            full_return_conf = port_info["full_return"]
            empty_return_conf = port_info["empty_return"]

            dist_conf = port_info["order_distribution"]
            source_dist_conf = dist_conf["source"]

            targets_dist = []

            # orders distribution to destination
            if "targets" in dist_conf:
                for target_port_name, target_conf in dist_conf["targets"].items():
                    dist = NoisedItem(
                        ports_mapping[target_port_name],
                        target_conf["proportion"],
                        target_conf["noise"])

                    targets_dist.append(dist)

            port_setting = SyntheticPortSetting(
                port_idx,
                port_name,
                port_info["capacity"],
                int(empty_ratio * total_container),
                NoisedItem(
                    port_idx,
                    empty_return_conf["buffer_ticks"],
                    empty_return_conf["noise"]),
                NoisedItem(
                    port_idx,
                    full_return_conf["buffer_ticks"],
                    full_return_conf["noise"]),
                NoisedItem(
                    port_idx,
                    source_dist_conf["proportion"],
                    source_dist_conf["noise"]),
                targets_dist)

            port_settings.append(port_setting)

        return ports_mapping, port_settings
