#Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import time
import sys
import random
import io
import os
import yaml

from maro.utils import convert_dottable
from maro.communication import Proxy, SessionType, SessionMessage

random.seed(2020)

CONFIG_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], "config.yml")
with io.open(CONFIG_PATH, "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

message_number = config.total_message
stop_msg_number = config.stop_message
redis_address = (config.redis.host_name, config.redis.port)

proxy = Proxy(group_name=config.group_name,
              component_type="learner",
              expected_peers=config.learner.peer,
              redis_address=redis_address,
              enable_rejoin=config.rejoin.enable,
              peer_update_frequency=config.rejoin.peers_update_frequency,
              minimal_peers=config.rejoin.minimal_peers,
              enable_message_cache_for_rejoin=config.rejoin.message_cache_for_rejoin,
              max_wait_time_for_rejoin=config.rejoin.max_wait_time_for_rejoin,
              log_enable=True)

peers = proxy.peers

# message list build
message_tag_list = ["cont"] * message_number
stop_tag_posi = random.sample(range(message_number), 2)
for s in stop_tag_posi:
    message_tag_list[s] = "stop"

for idx, msg_tag in enumerate(message_tag_list):
    time.sleep(random.randint(0,5))
    peer_idx = idx % len(peers["actor"])
    message = SessionMessage(tag=msg_tag,
                             source=proxy.component_name,
                             destination=peers["actor"][peer_idx],
                             payload=msg_tag,
                             session_type=SessionType.TASK)
    reply = proxy.send(message)
    if not isinstance(reply, list):
        print(reply)

proxy.ibroadcast(tag="finish", session_type=SessionType.TASK)
