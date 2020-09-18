# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import os
import subprocess
import sys

import yaml

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
sudo curl -L "https://github.com/docker/compose/releases/download/1.22.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose -s
sudo chmod +x /usr/local/bin/docker-compose
# newgrp docker : cannot use this command at here

# setup samba mount
echo 'setup samba mount'
mkdir -p {maro_path}
sudo mount -t cifs -o username={admin_username},password={samba_password} //{master_hostname}/sambashare {maro_path}
echo '//{master_hostname}/sambashare  {maro_path} cifs  username={admin_username},password={samba_password}  0  0' | sudo tee -a /etc/fstab

# load master public key
echo 'load master public key'
echo '{master_public_key}' >> ~/.ssh/authorized_keys

# delete outdated files
echo 'delete outdated files'
rm ~/details.yml
rm ~/init_node.py

# install pip3 and redis
echo 'install pip3 and redis'
sudo apt install -y python3-pip
pip3 install redis

echo "Finish node initialization"
'''

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_name')
    parser.add_argument('node_name')
    args = parser.parse_args()

    # Load details
    with open(os.path.expanduser(f"~/details.yml"), 'r') as fr:
        cluster_details = yaml.safe_load(fr)
    master_hostname = cluster_details['master']['hostname']
    master_public_key = cluster_details['master']['public_key']
    admin_username = cluster_details['user']['admin_username']
    samba_password = cluster_details['master']['samba']['password']

    # Load command
    command = INIT_COMMAND.format(
        admin_username=admin_username,
        maro_path=os.path.expanduser(f"~/.maro"),
        samba_password=samba_password,
        master_hostname=master_hostname,
        master_public_key=master_public_key,
    )

    # Exec command
    process = subprocess.Popen(command,
                               executable='/bin/bash',
                               shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8')
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
