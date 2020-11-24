# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.utils import convert_dottable

proxy = convert_dottable({
    "fault_tolerant": False,
    "delay_for_slow_joiner": 3,
    "redis": {
        "host": "localhost",
        "port": 6379,
        "max_retries": 10,
        "base_retry_interval": 0.1
    },
    "peer_rejoin": {
        "enable": False,
        "peers_catch_lifetime": 10,
        "minimal_peers": 1,  # int, minimal request peer number; or dict {"peer_type": int} for each peer type
        "enable_message_cache": False,
        "max_length_for_message_cache": 1024,   # The maximum number of pending messages for each peer
        "timeout_for_minimal_peer_number": 300,  # second
        "is_remove_failed_container": False,  # Automatically clean the failed container
        "max_rejoin_times": 5
    }
})

driver = convert_dottable({
    "zmq": {
        "protocol": "tcp",
        "send_timeout": -1,
        "receive_timeout": -1
    }
})
