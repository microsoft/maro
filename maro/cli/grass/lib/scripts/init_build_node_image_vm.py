# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import subprocess
import sys

INIT_COMMAND = """\
echo 'Step 1/{steps}: Install nvidia driver'
sudo apt-get install linux-headers-$(uname -r)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | tr -d '.')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-$distribution.pin
sudo mv cuda-$distribution.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/7fa2af80.pub
echo "deb http://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64 /" | \
    sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-get update
sudo apt-get -y install cuda-drivers

echo 'Step 2/{steps}: Install docker'
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce

echo 'Step 3/{steps}: Install nvidia container toolkit'
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

echo 'Step 4/{steps}: Install python3 and related packages'
sudo apt update
sudo apt install -y python3-pip
pip3 install redis

echo 'Step 5/{steps}: Delete outdated files'
rm ~/init_build_node_image_vm.py
"""

if __name__ == "__main__":
    # Exec command
    command = INIT_COMMAND.format(steps=5)
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
