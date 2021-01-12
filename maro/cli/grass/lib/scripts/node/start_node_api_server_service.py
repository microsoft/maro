# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import os
import sys
from pathlib import Path

from ..utils.details_reader import DetailsReader
from ..utils.subprocess import SubProcess

START_NODE_API_SERVER_COMMAND = """\
systemctl --user daemon-reload
systemctl --user start maro-node-api-server.service
systemctl --user enable maro-node-api-server.service
loginctl enable-linger {admin_username}  # Make sure the user is not logged out
"""

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_name")
    args = parser.parse_args()

    # Load details
    cluster_details = DetailsReader.load_cluster_details(cluster_name=args.cluster_name)
    admin_username = cluster_details["user"]["admin_username"]
    api_server_port = cluster_details["connection"]["api_server"]["port"]

    # Load .service
    with open(os.path.expanduser("~/.maro-shared/lib/grass/services/node_api_server/maro-node-api-server.service"), "r") as fr:
        service_file = fr.read()

    # Rewrite data in .service and write it to systemd folder
    service_file = service_file.format(home_path=str(Path.home()), api_server_port=api_server_port)
    os.makedirs(os.path.expanduser("~/.config/systemd/user/"), exist_ok=True)
    with open(os.path.expanduser("~/.config/systemd/user/maro-node-api-server.service"), "w") as fw:
        fw.write(service_file)

    # Exec command
    command = START_NODE_API_SERVER_COMMAND.format(admin_username=admin_username)
    return_str = SubProcess.run(command=command)
    sys.stdout.write(return_str)
