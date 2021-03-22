# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Init Build Node Image VM.

[WARNING] This script is a standalone script, which cannot use the ./utils tools.

Only executed in grass/azure mode.

After the initialization, this VM will deallocate and become a VM Image.
See https://docs.microsoft.com/en-us/azure/virtual-machines/linux/capture-image for reference.

The script will do the following jobs in this VM:
- Install docker.
- Install python3 and related packages.
- Install nvidia driver and nvidia container toolkit.
"""

import subprocess
import sys

INIT_RUNTIME_ENV_COMMAND = """\
# Set noninteractive to avoid irrelevant warning messages
export DEBIAN_FRONTEND=noninteractive

echo 'Step 1/{steps}: Install docker'
sudo -E apt-get update
sudo -E apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo -E apt-key add -
sudo -E apt-key fingerprint 0EBFCD88
sudo -E add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo -E apt-get update
sudo -E apt-get install -y docker-ce docker-ce-cli containerd.io

echo 'Step 2/{steps}: Install python3 and related packages'
sudo -E apt update
sudo -E apt install -y python3-pip
pip3 install redis psutil flask gunicorn pyyaml requests deepdiff

echo 'Step 3/{steps}: Install nvidia driver'
sudo -E apt-get install linux-headers-$(uname -r)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | tr -d '.')
wget --quiet https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-$distribution.pin
sudo -E mv cuda-$distribution.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo -E apt-key adv \
--fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/7fa2af80.pub
echo "deb http://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64 /" \
| sudo -E tee /etc/apt/sources.list.d/cuda.list
sudo -E apt-get update
sudo -E apt-get -y install cuda-drivers

echo 'Step 4/{steps}: Install nvidia container toolkit'
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo -E apt-key add - \
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
| sudo -E tee /etc/apt/sources.list.d/nvidia-docker.list
sudo -E apt-get update
sudo -E apt-get install -y nvidia-docker2
sudo -E systemctl restart docker

echo 'Step 5/{steps}: Delete outdated files'
rm ~/init_build_node_image_vm.py
"""

if __name__ == "__main__":
    # Parse and exec command
    command = INIT_RUNTIME_ENV_COMMAND.format(steps=5)
    process = subprocess.Popen(
        command,
        executable="/bin/bash",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    while True:
        next_line = process.stdout.readline()
        if next_line == "" and process.poll() is not None:
            break
        sys.stdout.write(next_line)
        sys.stdout.flush()
    stdout, stderr = process.communicate()
    if stderr:
        sys.stderr.write(stderr)
