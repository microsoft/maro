# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from ..utils.details_reader import DetailsReader
from ..utils.redis_controller import RedisController

# Details related

local_cluster_details = DetailsReader.load_local_cluster_details()
local_master_details = DetailsReader.load_local_master_details()

# Controllers related

redis_controller = RedisController(host="localhost", port=local_master_details["redis"]["port"])
