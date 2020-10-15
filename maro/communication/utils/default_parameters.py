# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.utils import convert_dottable


proxy = convert_dottable({
    "fault_tolerant": False,
    "delay_for_slow_joiner": 3,
    "peers_update_frequency": 10,   # unit is second.
    "dynamic_peer": False,
    "redis": {
        "host": "localhost",
        "port": 6379,
        "max_retries": 5,
        "base_retry_interval": 0.1
    }
})

driver = convert_dottable({
    "zmq": {
        "protocol": "tcp",
        "send_timeout": -1,
        "receive_timeout": -1
    }
})
