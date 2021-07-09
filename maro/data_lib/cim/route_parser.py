
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List

from .entities import RoutePoint


class RoutesParser:
    """Parser used to parse route information from configurations.
    """

    def parse(self, conf: dict) -> (Dict[str, int], List[List[RoutePoint]]):
        """Parse specified route configuration.

        Args:
            conf (dict): Configuration to parse.

        Returns:
            (Dict[str, int], List[List[RoutePoint]]): Route mapping (name to index),
                and list of route point list (index is route index).
        """
        routes: List[List[RoutePoint]] = []
        route_mapping: Dict[str, int] = {}  # name->idx

        idx = 0

        for name, node in conf.items():
            route_mapping[name] = idx

            routes.append([RoutePoint(idx, n["port_name"], n["distance_to_next_port"]) for n in node])

            idx += 1

        return route_mapping, routes
