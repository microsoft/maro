# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Stop the maro-node-agent service.
"""

from ..utils.subprocess import Subprocess

STOP_SERVICE_COMMAND = "systemctl --user stop maro-node-agent.service"

if __name__ == "__main__":
    # Stop service
    _ = Subprocess.run(command=STOP_SERVICE_COMMAND)
