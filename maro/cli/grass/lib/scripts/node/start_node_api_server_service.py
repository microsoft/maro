# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Start the maro-node-api-server service.
"""

import os
import pathlib
import pwd
import sys

from ..utils.details_reader import DetailsReader
from ..utils.params import Paths
from ..utils.subprocess import Subprocess

START_NODE_API_SERVER_COMMAND = """\
systemctl --user daemon-reload
systemctl --user start maro-node-api-server.service
systemctl --user enable maro-node-api-server.service
loginctl enable-linger {node_username}  # Make sure the user is not logged out
"""

if __name__ == "__main__":
    # Load details
    local_node_details = DetailsReader.load_local_node_details()

    # Rewrite data in .service and write it to systemd folder
    with open(
        file=f"{Paths.ABS_MARO_SHARED}/lib/grass/services/node_api_server/maro-node-api-server.service",
        mode="r"
    ) as fr:
        service_file = fr.read()
    service_file = service_file.format(
        home_path=str(pathlib.Path.home()),
        maro_shared_path=Paths.ABS_MARO_SHARED,
        node_api_server_port=local_node_details["api_server"]["port"]
    )
    os.makedirs(os.path.expanduser("~/.config/systemd/user/"), exist_ok=True)
    with open(file=os.path.expanduser("~/.config/systemd/user/maro-node-api-server.service"), mode="w") as fw:
        fw.write(service_file)

    # Exec command
    command = START_NODE_API_SERVER_COMMAND.format(node_username=pwd.getpwuid(os.getuid()).pw_name)
    return_str = Subprocess.run(command=command)
    sys.stdout.write(return_str)
