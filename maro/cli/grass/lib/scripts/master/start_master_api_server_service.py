# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import json
import os
import sys
from pathlib import Path

from ..utils.details import load_cluster_details
from ..utils.subprocess import SubProcess

START_MASTER_API_SERVER_COMMAND = """\
systemctl --user daemon-reload
systemctl --user start maro-master-api-server.service
systemctl --user enable maro-master-api-server.service
loginctl enable-linger {admin_username}  # Make sure the user is not logged out
"""

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_name")
    args = parser.parse_args()

    # Load details
    cluster_details = load_cluster_details(cluster_name=args.cluster_name)
    admin_username = cluster_details["user"]["admin_username"]
    redis_port = cluster_details["master"]["redis"]["port"]
    api_server_port = cluster_details["connection"]["api_server"]["port"]

    # Dump master_agent.config
    os.makedirs(os.path.expanduser("~/.maro-local/services/"), exist_ok=True)
    with open(os.path.expanduser("~/.maro-local/services/maro-master-api-server.config"), "w") as fw:
        json.dump(
            obj={
                "cluster_name": args.cluster_name,
                "redis_port": redis_port
            },
            fp=fw
        )

    # Load .service
    with open(
        os.path.expanduser("~/.maro/lib/grass/services/master_api_server/maro-master-api-server.service"), "r"
    ) as fr:
        service_file = fr.read()

    # Rewrite data in .service and write it to systemd folder
    service_file = service_file.format(home_path=str(Path.home()), api_server_port=api_server_port)
    os.makedirs(os.path.expanduser("~/.config/systemd/user/"), exist_ok=True)
    with open(os.path.expanduser("~/.config/systemd/user/maro-master-api-server.service"), "w") as fw:
        fw.write(service_file)

    # Exec command
    command = START_MASTER_API_SERVER_COMMAND.format(admin_username=admin_username)
    return_str = SubProcess.run(command=command)
    sys.stdout.write(return_str)
