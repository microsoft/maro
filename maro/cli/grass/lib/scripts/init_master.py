# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import os
import subprocess
import sys

from .utils import load_cluster_details

INIT_COMMAND = '''\
# create group 'docker' and add admin user
sudo groupadd docker
sudo gpasswd -a {admin_username} docker

# install docker
echo 'install docker'
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce
sudo curl -L "https://github.com/docker/compose/releases/download/1.22.0/docker-compose-$(uname -s)-$(uname -m)"\
    -o /usr/local/bin/docker-compose -s
sudo chmod +x /usr/local/bin/docker-compose
# newgrp docker : cannot use this command at here

# install and launch redis
echo 'install and launch redis'
sudo docker pull redis
sudo docker run -p {redis_port}:6379 -v ~/.maro/lib/grass/redis/redis.conf:/maro/lib/grass/redis/redis.conf\
    --name maro-redis -d redis redis-server /maro/lib/grass/redis/redis.conf

# install and launch samba
echo 'install and launch samba'
sudo apt install -y samba
echo -e "[sambashare]\n    comment = Samba on Ubuntu\n    path = {maro_path}\n    read only = no\n    browsable = yes"\
    | sudo tee -a /etc/samba/smb.conf
sudo service smbd restart
sudo ufw allow samba
(echo "{samba_password}"; echo "{samba_password}") | sudo smbpasswd -a {admin_username}

# install and launch fluentd
echo 'install and launch samba'
sudo docker pull fluent/fluentd
sudo docker run -p {fluentd_port}:24224 -v ~/.maro/logs:/fluentd/log\
    -v ~/.maro/lib/grass/fluentd/fluentd.conf:/fluentd/etc/fluentd.conf\
    -e FLUENTD_CONF=fluentd.conf --name maro-fluentd -d fluent/fluentd

# install pip3 and redis
echo 'install pip3 and redis'
sudo apt install -y python3-pip
pip3 install redis

echo "Finish master initialization"
'''

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_name')
    args = parser.parse_args()

    # Load details
    cluster_details = load_cluster_details(cluster_name=args.cluster_name)
    admin_username = cluster_details['user']['admin_username']
    samba_password = cluster_details['master']['samba']['password']
    redis_port = cluster_details['master']['redis']['port']
    fluentd_port = cluster_details['master']['fluentd']['port']

    # Parse command
    command = INIT_COMMAND.format(
        admin_username=admin_username,
        samba_password=samba_password,
        maro_path=os.path.expanduser("~/.maro"),
        redis_port=redis_port,
        fluentd_port=fluentd_port
    )

    # Exec command
    process = subprocess.Popen(
        command, executable='/bin/bash', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
    )
    while True:
        nextline = process.stdout.readline()
        if nextline == '' and process.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()
    stdout, stderr = process.communicate()
    if stderr:
        sys.stderr.write(stderr.strip('\n'))
    sys.stdout.write(stdout.strip('\n'))
