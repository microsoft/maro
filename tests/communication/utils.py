# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket


def get_random_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
        temp_socket.bind(("", 0))
        random_port = temp_socket.getsockname()[1]
    
    return random_port
