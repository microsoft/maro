# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Init MARO Master VM,

The script will do the following jobs in this VM:
- Install docker,
- Start maro-redis and maro-fluentd,
- Start samba server,
- Install python3 and related packages.
- Start MARO Master services: maro-master-agent and maro-master-api-server.
"""

import argparse
import os
import pathlib
import pwd
import shutil

from ..utils.details_reader import DetailsReader
from ..utils.details_writer import DetailsWriter
from ..utils.params import Paths
from ..utils.subprocess import Subprocess

INIT_COMMAND = """\
# Set noninteractive to avoid irrelevant warning messages
export DEBIAN_FRONTEND=noninteractive

# create group 'docker' and add admin user
sudo -E groupadd docker
sudo -E gpasswd -a {master_username} docker

# install docker
echo 'Step 1/{steps}: Install docker'
sudo -E apt-get update
sudo -E apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo -E apt-key add -
sudo -E apt-key fingerprint 0EBFCD88
sudo -E add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo -E apt-get update
sudo -E apt-get install -y docker-ce docker-ce-cli containerd.io

# install and launch redis
echo 'Step 2/{steps}: Install and launch redis'
sudo -E docker pull redis
sudo -E docker run -p {master_redis_port}:6379 \
-v {maro_shared_path}/lib/grass/configs/redis/redis.conf:/maro/lib/grass/redis/redis.conf \
--name maro-redis-{cluster_id} -d redis redis-server /maro/lib/grass/redis/redis.conf

# install and launch samba
echo 'Step 3/{steps}: Install and launch samba'
sudo -E apt install -y samba
echo -e "[sambashare]\n    \
comment = Samba for MARO\n    \
path = {maro_shared_path}\n    \
read only = no\n    \
browsable = yes" \
| sudo -E tee -a /etc/samba/smb.conf
sudo -E service smbd restart
sudo -E ufw allow samba
(echo "{master_samba_password}"; echo "{master_samba_password}") | sudo -E smbpasswd -a {master_username}

# install and launch fluentd
echo 'Step 4/{steps}: Install and launch fluentd'
sudo -E docker pull fluent/fluentd
sudo -E docker run -p {master_fluentd_port}:24224 -v {maro_shared_path}/clusters/{cluster_name}/logs:/fluentd/log \
-v {maro_shared_path}/lib/grass/configs/fluentd/fluentd.conf:/fluentd/etc/fluentd.conf \
-e FLUENTD_CONF=fluentd.conf --name maro-fluentd-{cluster_id} -d fluent/fluentd

# install pip3 and redis
echo 'Step 5/{steps}: Install pip3 and redis'
sudo -E apt install -y python3-pip
pip3 install --upgrade redis flask gunicorn cryptography pyjwt

echo "Finish master initialization"
"""

START_MASTER_AGENT_COMMAND = """\
systemctl --user daemon-reload
systemctl --user start maro-master-agent.service
systemctl --user enable maro-master-agent.service
loginctl enable-linger {master_username}  # Make sure the user is not logged out
"""

START_MASTER_API_SERVER_COMMAND = """\
systemctl --user daemon-reload
systemctl --user start maro-master-api-server.service
systemctl --user enable maro-master-api-server.service
loginctl enable-linger {master_username}  # Make sure the user is not logged out
"""


class MasterInitializer:
    def __init__(self, cluster_details: dict):
        self.cluster_details = cluster_details
        self.master_details = cluster_details["master"]

    def init_master(self):
        # Parse and exec command
        command = INIT_COMMAND.format(
            cluster_id=self.cluster_details["id"],
            master_username=pwd.getpwuid(os.getuid()).pw_name,
            master_samba_password=self.master_details["samba"]["password"],
            maro_shared_path=Paths.ABS_MARO_SHARED,
            master_redis_port=self.master_details["redis"]["port"],
            cluster_name=cluster_details["name"],
            master_fluentd_port=self.master_details["fluentd"]["port"],
            steps=5
        )
        Subprocess.interactive_run(command=command)

    @staticmethod
    def start_master_agent():
        # Rewrite data in .service and write it to systemd folder
        with open(
            file=f"{Paths.ABS_MARO_SHARED}/lib/grass/services/master_agent/maro-master-agent.service",
            mode="r"
        ) as fr:
            service_file = fr.read()
        service_file = service_file.format(maro_shared_path=Paths.ABS_MARO_SHARED)
        os.makedirs(name=os.path.expanduser("~/.config/systemd/user/"), exist_ok=True)
        with open(file=os.path.expanduser("~/.config/systemd/user/maro-master-agent.service"), mode="w") as fw:
            fw.write(service_file)

        # Parse and exec command
        command = START_MASTER_AGENT_COMMAND.format(master_username=pwd.getpwuid(os.getuid()).pw_name)
        Subprocess.interactive_run(command=command)

    def start_master_api_server(self):
        # Rewrite data in .service and write it to systemd folder
        with open(
            file=f"{Paths.ABS_MARO_SHARED}/lib/grass/services/master_api_server/maro-master-api-server.service",
            mode="r"
        ) as fr:
            service_file = fr.read()
        service_file = service_file.format(
            home_path=str(pathlib.Path.home()),
            maro_shared_path=Paths.ABS_MARO_SHARED,
            master_api_server_port=self.master_details["api_server"]["port"]
        )
        os.makedirs(name=os.path.expanduser("~/.config/systemd/user/"), exist_ok=True)
        with open(file=os.path.expanduser("~/.config/systemd/user/maro-master-api-server.service"), mode="w") as fw:
            fw.write(service_file)

        # Parse and exec command
        command = START_MASTER_API_SERVER_COMMAND.format(master_username=pwd.getpwuid(os.getuid()).pw_name)
        Subprocess.interactive_run(command=command)

    @staticmethod
    def copy_scripts():
        os.makedirs(name=f"{Paths.ABS_MARO_LOCAL}/scripts", exist_ok=True)
        shutil.copy2(
            src=f"{Paths.ABS_MARO_SHARED}/lib/grass/scripts/master/delete_master.py",
            dst=f"{Paths.ABS_MARO_LOCAL}/scripts"
        )


if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_name")
    args = parser.parse_args()

    # Load details
    cluster_details = DetailsReader.load_cluster_details(cluster_name=args.cluster_name)

    # Save local details
    DetailsWriter.save_local_cluster_details(cluster_details=cluster_details)
    DetailsWriter.save_local_master_details(master_details=cluster_details["master"])

    # Start initializing
    master_initializer = MasterInitializer(cluster_details=cluster_details)
    master_initializer.init_master()
    master_initializer.start_master_agent()
    master_initializer.start_master_api_server()

    # Copy scripts
    master_initializer.copy_scripts()
