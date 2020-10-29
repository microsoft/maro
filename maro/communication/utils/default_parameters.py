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
        "enable_message_cache": True,
        "timeout_for_minimal_peer_number": 300,  # second
        "auto_clean_for_container": False,
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
