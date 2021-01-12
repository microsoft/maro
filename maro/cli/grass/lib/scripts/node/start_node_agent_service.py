# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import os
import pwd
import sys
from pathlib import Path

from ..utils.details_reader import DetailsReader
from ..utils.subprocess import SubProcess

START_NODE_AGENT_COMMAND = """\
systemctl --user daemon-reload
systemctl --user start maro-node-agent.service
systemctl --user enable maro-node-agent.service
loginctl enable-linger {node_username}  # Make sure the user is not logged out
"""

if __name__ == "__main__":
    # Load details
    cluster_details = DetailsReader.load_local_cluster_details()
    node_details = DetailsReader.load_local_node_details()

    # Dump node_agent.config
    os.makedirs(os.path.expanduser("~/.maro-local/services/"), exist_ok=True)
    with open(os.path.expanduser("~/.maro-local/services/maro-node-agent.config"), "w") as fw:
        json.dump({
            "cluster_name": cluster_details["name"],
            "node_name": node_details["name"],
            "master_hostname": cluster_details["master"]["hostname"],
            "master_redis_port": cluster_details["master"]["redis"]["port"]
        }, fw)

    # Load .service
    with open(os.path.expanduser("~/.maro-shared/lib/grass/services/node_agent/maro-node-agent.service"), "r") as fr:
        service_file = fr.read()

    # Rewrite data in .service and write it to systemd folder
    service_file = service_file.format(home_path=str(Path.home()))
    os.makedirs(os.path.expanduser("~/.config/systemd/user/"), exist_ok=True)
    with open(os.path.expanduser("~/.config/systemd/user/maro-node-agent.service"), "w") as fw:
        fw.write(service_file)

    # Exec command
    command = START_NODE_AGENT_COMMAND.format(node_username=pwd.getpwuid(os.getuid()).pw_name)
    return_str = SubProcess.run(command=command)
    sys.stdout.write(return_str)
