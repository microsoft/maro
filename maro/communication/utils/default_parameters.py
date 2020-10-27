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
        "peers_update_frequency": 10,
        "minimal_peers": 1,  # int, minimal request peer number; or dict {"peer_type": int} for each peer type
        "enable_message_cache": True,
        "max_wait_time_for_rejoin": 300,  # second
        "auto_clean_for_container": False
    }
})

driver = convert_dottable({
    "zmq": {
        "protocol": "tcp",
        "send_timeout": -1,
        "receive_timeout": -1
    }
})
