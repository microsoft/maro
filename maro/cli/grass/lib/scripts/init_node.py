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

# setup samba mount
echo 'Step 1/{steps}: Setup samba mount'
mkdir -p {maro_path}
sudo mount -t cifs -o username={admin_username},password={samba_password} //{master_hostname}/sambashare {maro_path}
echo '//{master_hostname}/sambashare  {maro_path} cifs  username={admin_username},password={samba_password}  0  0' | \
    sudo tee -a /etc/fstab

# load master public key
echo 'Step 2/{steps}: Load master public key'
echo '{master_public_key}' >> ~/.ssh/authorized_keys

# delete outdated files
echo 'Step 3/{steps}: Delete outdated files'
rm ~/details.yml
rm ~/init_node.py

echo "Finish node initialization"
'''

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_name")
    parser.add_argument("node_name")
    args = parser.parse_args()

    # Load details
    with open(os.path.expanduser("~/details.yml"), "r") as fr:
        cluster_details = yaml.safe_load(fr)
    master_hostname = cluster_details["master"]["hostname"]
    master_public_key = cluster_details["master"]["public_key"]
    admin_username = cluster_details["user"]["admin_username"]
    samba_password = cluster_details["master"]["samba"]["password"]

    # Load command
    command = INIT_COMMAND.format(
        admin_username=admin_username,
        maro_path=os.path.expanduser("~/.maro"),
        samba_password=samba_password,
        master_hostname=master_hostname,
        master_public_key=master_public_key,
        steps=3
    )

    # Exec command
    process = subprocess.Popen(
        command,
        executable="/bin/bash",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
    )
    while True:
        nextline = process.stdout.readline()
        if nextline == "" and process.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()
    stdout, stderr = process.communicate()
    if stderr:
        sys.stderr.write(stderr.strip("\n"))
    sys.stdout.write(stdout.strip("\n"))
