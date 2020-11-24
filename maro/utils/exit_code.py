# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


# EXIT CODE
NON_RESTART_EXIT_CODE = 64  # If a container exited with the code 64, do not restart it.
KILL_ALL_EXIT_CODE = 65     # If a container exited with the code 65, kill all containers with the same job_id.
