# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket

from maro.communication import Proxy


def get_random_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
        temp_socket.bind(("", 0))
        random_port = temp_socket.getsockname()[1]

    return random_port


def proxy_generator(component_type, redis_port):
    proxy_parameters = {
        "group_name": "communication_unit_test",
        "redis_address": ("localhost", redis_port),
        "log_enable": False
    }

    component_type_expected_peers_map = {
        "receiver": {"sender": 1},
        "sender": {"receiver": 1},
        "master": {"worker": 5},
        "worker": {"master": 1}
    }

    proxy = Proxy(
        component_type=component_type,
        expected_peers=component_type_expected_peers_map[component_type],
        **proxy_parameters
    )

    return proxy
