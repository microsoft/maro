# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import os

from ..utils.redis_controller import RedisController

# Config related

with open(os.path.expanduser("~/.maro-local/services/maro-master-api-server.config"), "r") as fr:
    service_config = json.load(fr)

# Controllers related

redis_controller = RedisController(host="localhost", port=service_config["master_redis_port"])
