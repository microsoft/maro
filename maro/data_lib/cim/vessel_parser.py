# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List

from .entities import VesselSetting


class VesselsParser:
    """Parser used to parse vessel configurations.
    """

    def parse(self, conf: dict) -> (Dict[str, int], List[VesselSetting]):
        """Parse specified vessel configurations.

        Args:
            conf(dict): Configurations to parse.

        Returns:
            (Dict[str, int], List[VesselSetting]): Vessel mappings (name to index), and settings list for all vessels.

        """
        mapping: Dict[str, int] = {}
        vessels: List[VesselSetting] = []

        index = 0

        for vessel_name, vessel_node in conf.items():
            mapping[vessel_name] = index

            sailing = vessel_node["sailing"]
            parking = vessel_node["parking"]
            route = vessel_node["route"]

            vessels.append(VesselSetting(
                index,
                vessel_name,
                vessel_node["capacity"],
                route["route_name"],
                route["initial_port_name"],
                sailing["speed"],
                sailing["noise"],
                parking["duration"],
                parking["noise"],
                # default no empty
                vessel_node.get("empty", 0)))

            index += 1

        return mapping, vessels
