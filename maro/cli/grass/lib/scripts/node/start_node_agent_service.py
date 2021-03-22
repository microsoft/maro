# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Start the maro-node-agent service.
"""

import os
import pwd
import sys

from ..utils.params import Paths
from ..utils.subprocess import Subprocess

START_NODE_AGENT_COMMAND = """\
systemctl --user daemon-reload
systemctl --user start maro-node-agent.service
systemctl --user enable maro-node-agent.service
loginctl enable-linger {node_username}  # Make sure the user is not logged out
"""

if __name__ == "__main__":
    # Rewrite data in .service and write it to systemd folder
    with open(
        file=f"{Paths.ABS_MARO_SHARED}/lib/grass/services/node_agent/maro-node-agent.service",
        mode="r"
    ) as fr:
        service_file = fr.read()
    service_file = service_file.format(maro_shared_path=Paths.ABS_MARO_SHARED)
    os.makedirs(name=os.path.expanduser("~/.config/systemd/user/"), exist_ok=True)
    with open(file=os.path.expanduser("~/.config/systemd/user/maro-node-agent.service"), mode="w") as fw:
        fw.write(service_file)

    # Exec command
    command = START_NODE_AGENT_COMMAND.format(node_username=pwd.getpwuid(os.getuid()).pw_name)
    return_str = Subprocess.run(command=command)
    sys.stdout.write(return_str)
