# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from ..utils.subprocess import SubProcess

STOP_SERVICE_COMMAND = "systemctl --user stop maro-node-agent.service"

if __name__ == "__main__":
    # Stop service
    _ = SubProcess.run(command=STOP_SERVICE_COMMAND)
