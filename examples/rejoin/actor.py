# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import numpy as np
import time
import sys
import io
import os
import yaml

from maro.utils import convert_dottable
from maro.communication import Proxy, SessionType, SessionMessage


CONFIG_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], "config.yml")

with io.open(CONFIG_PATH, "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

redis_address = (config.redis.host_name, config.redis.port)

proxy = Proxy(group_name=config.group_name,
              component_type="actor",
              expected_peers=config.actor.peer,
              redis_address=redis_address,
              enable_rejoin=config.rejoin.enable,
              peer_update_frequency=config.rejoin.peers_update_frequency,
              minimal_peers=config.rejoin.minimal_peers,
              enable_message_cache_for_rejoin=config.rejoin.message_cache_for_rejoin,
              max_wait_time_for_rejoin=config.rejoin.max_wait_time_for_rejoin,
              log_enable=True)

# continuously receive messages from proxy
for msg in proxy.receive(is_continuous=True):
    if msg.tag == "cont":
        proxy.reply(received_message=msg, tag="recv", payload="successful receive!")
    elif msg.tag == "stop":
        proxy.reply(received_message=msg, tag="recv", payload=f"{proxy.component_name} exited!")
        sys.exit(1)
    elif msg.tag == "finish":
        proxy.reply(received_message=msg, tag="recv", payload=f"{proxy.component_name} finish!")
        sys.exit(0)
