# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


process_setting = {
    "redis_info": {
        "host": "localhost",
        "port": 19999
    },
    "parallel_level": 1,
    "keep_agent_alive": 1,  # If 0 (False), agents will exit after 5 minutes of no pending jobs and running jobs.
    "redis_mode": "MARO"    # one of MARO, customized. customized Redis won't exit after maro process clear.
}
