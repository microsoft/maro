# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import os
import pwd

from ..utils.details_reader import DetailsReader
from ..utils.subprocess import SubProcess

INIT_COMMAND = """\
# create group 'docker' and add admin user
sudo groupadd docker
sudo gpasswd -a {master_username} docker

# install docker
echo 'Step 1/{steps}: Install docker'
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
# newgrp docker : cannot use this command at here

# install and launch redis
echo 'Step 2/{steps}: Install and launch redis'
sudo docker pull redis
sudo docker run -p {redis_port}:6379 -v ~/.maro-shared/lib/grass/configs/redis/redis.conf:/maro/lib/grass/redis/redis.conf\
    --name maro-redis -d redis redis-server /maro/lib/grass/redis/redis.conf

# install and launch samba
echo 'Step 3/{steps}: Install and launch samba'
sudo apt install -y samba
echo -e "[sambashare]\n    comment = Samba on Ubuntu\n    path = {maro_shared_path}\n    read only = no\n    browsable = yes"\
    | sudo tee -a /etc/samba/smb.conf
sudo service smbd restart
sudo ufw allow samba
(echo "{samba_password}"; echo "{samba_password}") | sudo smbpasswd -a {master_username}

# install and launch fluentd
echo 'Step 4/{steps}: Install and launch fluentd'
sudo docker pull fluent/fluentd
sudo docker run -p {fluentd_port}:24224 -v ~/.maro-shared/logs:/fluentd/log\
    -v ~/.maro-shared/lib/grass/configs/fluentd/fluentd.conf:/fluentd/etc/fluentd.conf\
    -e FLUENTD_CONF=fluentd.conf --name maro-fluentd -d fluent/fluentd

# install pip3 and redis
echo 'Step 5/{steps}: Install pip3 and redis'
sudo apt install -y python3-pip
pip3 install redis flask gunicorn

echo "Finish master initialization"
"""

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_name")
    args = parser.parse_args()

    # Load details
    cluster_details = DetailsReader.load_cluster_details(cluster_name=args.cluster_name)
    samba_password = cluster_details["master"]["samba"]["password"]
    redis_port = cluster_details["master"]["redis"]["port"]
    fluentd_port = cluster_details["master"]["fluentd"]["port"]

    # Parse and exec command
    command = INIT_COMMAND.format(
        master_username=pwd.getpwuid(os.getuid()).pw_name,
        samba_password=samba_password,
        maro_shared_path=os.path.expanduser("~/.maro-shared"),
        redis_port=redis_port,
        fluentd_port=fluentd_port,
        steps=5
    )
    SubProcess.interactive_run(command=command)
