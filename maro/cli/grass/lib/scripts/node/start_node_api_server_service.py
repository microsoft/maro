# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import pwd
import sys
from pathlib import Path

from ..utils.details_reader import DetailsReader
from ..utils.subprocess import SubProcess

START_NODE_API_SERVER_COMMAND = """\
systemctl --user daemon-reload
systemctl --user start maro-node-api-server.service
systemctl --user enable maro-node-api-server.service
loginctl enable-linger {node_username}  # Make sure the user is not logged out
"""

if __name__ == "__main__":
    # Load details
    node_details = DetailsReader.load_local_node_details()

    # Load .service
    with open(
        file=os.path.expanduser("~/.maro-shared/lib/grass/services/node_api_server/maro-node-api-server.service"),
        mode="r"
    ) as fr:
        service_file = fr.read()

    # Rewrite data in .service and write it to systemd folder
    service_file = service_file.format(
        home_path=str(Path.home()),
        node_api_server_port=node_details["api_server"]["port"]
    )
    os.makedirs(os.path.expanduser("~/.config/systemd/user/"), exist_ok=True)
    with open(os.path.expanduser("~/.config/systemd/user/maro-node-api-server.service"), "w") as fw:
        fw.write(service_file)

    # Exec command
    command = START_NODE_API_SERVER_COMMAND.format(node_username=pwd.getpwuid(os.getuid()).pw_name)
    return_str = SubProcess.run(command=command)
    sys.stdout.write(return_str)
