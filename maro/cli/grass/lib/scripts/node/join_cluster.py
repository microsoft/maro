# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Join the current VM to the cluster.

[WARNING] This script is a standalone script, which cannot use the ./utils tools.

The script will do the following jobs in this VM:
- Install Node Runtime. [Optional, depend on join_cluster_deployment]
- Install Node GPU Support. [Optional, depend on join_cluster_deployment]
- Start MARO Node services: maro-node-agent, maro-node-api-server.
- Create docker user.
- Setup samba client to mount to the MARO Master.
- Add MARO Master's ssh_public_key to the current VM.
"""

import argparse
import functools
import json
import operator
import os
import pathlib
import pwd
import shutil
import subprocess
import sys
import uuid

import deepdiff
import redis
import yaml
from redis.lock import Lock

# Commands.

INSTALL_NODE_RUNTIME_COMMAND = """\
# Set noninteractive to avoid irrelevant warning messages
export DEBIAN_FRONTEND=noninteractive

echo 'Step 1/2: Install docker'
sudo -E apt-get update
sudo -E apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo -E apt-key add -
sudo -E apt-key fingerprint 0EBFCD88
sudo -E add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo -E apt-get update
sudo -E apt-get install -y docker-ce docker-ce-cli containerd.io

echo 'Step 2/2: Install python3 and related packages'
sudo -E apt update
sudo -E apt install -y python3-pip
pip3 install redis psutil flask gunicorn pyyaml requests deepdiff
"""

INSTALL_NODE_GPU_SUPPORT_COMMAND = """\
echo 'Step 1/2: Install nvidia driver'
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

echo 'Step 2/2: Install nvidia container toolkit'
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo -E apt-key add - \
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
| sudo -E tee /etc/apt/sources.list.d/nvidia-docker.list
sudo -E apt-get update
sudo -E apt-get install -y nvidia-docker2
sudo -E systemctl restart docker
"""

CREATE_DOCKER_USER_COMMAND = """\
sudo groupadd docker
sudo gpasswd -a {node_username} docker
"""

SETUP_SAMBA_MOUNT_COMMAND = """\
mkdir -p {maro_shared_path}
sudo mount -t cifs \
-o username={master_username},password={master_samba_password} //{master_hostname}/sambashare {maro_shared_path}
echo '//{master_hostname}/sambashare  \
{maro_shared_path} cifs  username={master_username},password={master_samba_password}  0  0' \
| sudo tee -a /etc/fstab
"""

START_NODE_AGENT_SERVICE_COMMAND = """\
systemctl --user daemon-reload
systemctl --user start maro-node-agent.service
systemctl --user enable maro-node-agent.service
loginctl enable-linger {node_username}  # Make sure the user is not logged out
"""

START_NODE_API_SERVER_SERVICE_COMMAND = """\
systemctl --user daemon-reload
systemctl --user start maro-node-api-server.service
systemctl --user enable maro-node-api-server.service
loginctl enable-linger {node_username}  # Make sure the user is not logged out
"""

APPEND_AUTHORIZED_KEY = """\
echo "{public_key}" >> ~/.ssh/authorized_keys
"""


# Node Joiner.

class NodeJoiner:
    def __init__(self, join_cluster_deployment: dict):
        self.join_cluster_deployment = join_cluster_deployment
        self.node_details = self.join_cluster_deployment["node"]

        redis_controller = RedisController(
            host=join_cluster_deployment["master"]["private_ip_address"],
            port=join_cluster_deployment["master"]["redis"]["port"]
        )
        self.cluster_details = redis_controller.get_cluster_details()

        self.node_details = self._init_node_details(node_details=self.node_details)
        with redis_controller.lock(f"lock:name_to_node_details:{self.node_details['name']}"):
            redis_controller.set_node_details(node_details=self.node_details)

        self.master_details = redis_controller.get_master_details()

    @staticmethod
    def _init_node_details(node_details: dict):
        # Init runtime params.
        if "name" not in node_details and "id" not in node_details:
            node_name = NameCreator.create_node_name()
            node_details["name"] = node_name
            node_details["id"] = node_name
        node_details["image_files"] = {}
        node_details["containers"] = {}
        node_details["state"] = {
            "status": NodeStatus.PENDING
        }

        return node_details

    def init_node_runtime_env(self):
        if self.join_cluster_deployment["configs"]["install_node_gpu_support"]:
            command = INSTALL_NODE_RUNTIME_COMMAND.format()
            Subprocess.interactive_run(command=command)
            command = INSTALL_NODE_GPU_SUPPORT_COMMAND.format()
            Subprocess.interactive_run(command=command)
        elif self.join_cluster_deployment["configs"]["install_node_runtime"]:
            command = INSTALL_NODE_RUNTIME_COMMAND.format()
            Subprocess.interactive_run(command=command)

    @staticmethod
    def create_docker_user():
        command = CREATE_DOCKER_USER_COMMAND.format(node_username=pwd.getpwuid(os.getuid()).pw_name)
        Subprocess.run(command=command)

    def setup_samba_mount(self):
        command = SETUP_SAMBA_MOUNT_COMMAND.format(
            master_username=self.master_details["username"],
            master_hostname=self.master_details["private_ip_address"],
            master_samba_password=self.master_details["samba"]["password"],
            maro_shared_path=Paths.ABS_MARO_SHARED
        )
        Subprocess.run(command=command)

    def start_node_agent_service(self):
        # Rewrite data in .service and write it to systemd folder.
        with open(
            file=f"{Paths.ABS_MARO_SHARED}/lib/grass/services/node_agent/maro-node-agent.service",
            mode="r"
        ) as fr:
            service_file = fr.read()
        service_file = service_file.format(maro_shared_path=Paths.ABS_MARO_SHARED)
        os.makedirs(name=os.path.expanduser("~/.config/systemd/user/"), exist_ok=True)
        with open(file=os.path.expanduser("~/.config/systemd/user/maro-node-agent.service"), mode="w") as fw:
            fw.write(service_file)

        command = START_NODE_AGENT_SERVICE_COMMAND.format(node_username=pwd.getpwuid(os.getuid()).pw_name)
        Subprocess.run(command=command)

    def start_node_api_server_service(self):
        # Rewrite data in .service and write it to systemd folder.
        with open(
            file=f"{Paths.ABS_MARO_SHARED}/lib/grass/services/node_api_server/maro-node-api-server.service",
            mode="r"
        ) as fr:
            service_file = fr.read()
        service_file = service_file.format(
            home_path=str(pathlib.Path.home()),
            maro_shared_path=Paths.ABS_MARO_SHARED,
            node_api_server_port=self.node_details["api_server"]["port"]
        )
        os.makedirs(os.path.expanduser("~/.config/systemd/user/"), exist_ok=True)
        with open(file=os.path.expanduser("~/.config/systemd/user/maro-node-api-server.service"), mode="w") as fw:
            fw.write(service_file)

        command = START_NODE_API_SERVER_SERVICE_COMMAND.format(node_username=pwd.getpwuid(os.getuid()).pw_name)
        Subprocess.run(command=command)

    @staticmethod
    def copy_leave_script():
        src_files = [
            f"{Paths.ABS_MARO_SHARED}/lib/grass/scripts/node/leave_cluster.py",
            f"{Paths.ABS_MARO_SHARED}/lib/grass/scripts/node/activate_leave_cluster.py"
        ]
        os.makedirs(name=f"{Paths.ABS_MARO_LOCAL}/scripts", exist_ok=True)
        for src_file in src_files:
            shutil.copy2(
                src=src_file,
                dst=f"{Paths.ABS_MARO_LOCAL}/scripts"
            )

    def load_master_public_key(self):
        command = APPEND_AUTHORIZED_KEY.format(public_key=self.master_details["ssh"]["public_key"])
        Subprocess.run(command=command)

    # Utils methods.

    @staticmethod
    def standardize_join_cluster_deployment(join_cluster_deployment: dict) -> dict:
        join_cluster_deployment_template = {
            "mode": "",
            "master": {
                "private_ip_address": "",
                "api_server": {
                    "port": ""
                },
                "redis": {
                    "port": ""
                }
            },
            "node": {
                "hostname": "",
                "username": "",
                "public_ip_address": "",
                "private_ip_address": "",
                "resources": {
                    "cpu": "",
                    "memory": "",
                    "gpu": ""
                },
                "ssh": {
                    "port": ""
                },
                "api_server": {
                    "port": ""
                }
            },
            "configs": {
                "install_node_runtime": "",
                "install_node_gpu_support": ""
            }
        }
        DeploymentValidator.validate_and_fill_dict(
            template_dict=join_cluster_deployment_template,
            actual_dict=join_cluster_deployment,
            optional_key_to_value={
                "root['master']['redis']": {"port": Params.DEFAULT_REDIS_PORT},
                "root['master']['redis']['port']": Params.DEFAULT_REDIS_PORT,
                "root['master']['api_server']": {"port": Params.DEFAULT_API_SERVER_PORT},
                "root['master']['api_server']['port']": Params.DEFAULT_API_SERVER_PORT,
                "root['node']['resources']": {
                    "cpu": "all",
                    "memory": "all",
                    "gpu": "all"
                },
                "root['node']['resources']['cpu']": "all",
                "root['node']['resources']['memory']": "all",
                "root['node']['resources']['gpu']": "all",
                "root['node']['api_server']": {"port": Params.DEFAULT_API_SERVER_PORT},
                "root['node']['api_server']['port']": Params.DEFAULT_API_SERVER_PORT,
                "root['node']['ssh']": {"port": Params.DEFAULT_SSH_PORT},
                "root['node']['ssh']['port']": Params.DEFAULT_SSH_PORT,
                "root['configs']": {
                    "install_node_runtime": False,
                    "install_node_gpu_support": False
                },
                "root['configs']['install_node_runtime']": False,
                "root['configs']['install_node_gpu_support']": False
            }
        )

        return join_cluster_deployment


# Utils Classes.

class Params:
    DEFAULT_SSH_PORT = 22
    DEFAULT_REDIS_PORT = 6379
    DEFAULT_API_SERVER_PORT = 51812


class NodeStatus:
    PENDING = "Pending"
    RUNNING = "Running"
    STOPPED = "Stopped"


class Paths:
    MARO_SHARED = "~/.maro-shared"
    ABS_MARO_SHARED = os.path.expanduser(MARO_SHARED)

    MARO_LOCAL = "~/.maro-local"
    ABS_MARO_LOCAL = os.path.expanduser(MARO_LOCAL)


class NameCreator:
    @staticmethod
    def create_name_with_uuid(prefix: str, uuid_len: int = 16) -> str:
        postfix = uuid.uuid4().hex[:uuid_len]
        return f"{prefix}{postfix}"

    @staticmethod
    def create_node_name():
        return NameCreator.create_name_with_uuid(prefix="node", uuid_len=8)


class RedisController:
    def __init__(self, host: str, port: int):
        self._redis = redis.Redis(host=host, port=port, encoding="utf-8", decode_responses=True)

    """Cluster Details Related."""

    def get_cluster_details(self) -> dict:
        return json.loads(self._redis.get("cluster_details"))

    """Master Details Related."""

    def get_master_details(self) -> dict:
        return json.loads(self._redis.get("master_details"))

    """Node Details Related."""

    def set_node_details(self, node_details: dict) -> None:
        self._redis.hset(
            "name_to_node_details",
            node_details["name"],
            json.dumps(node_details)
        )

    """Utils."""

    def lock(self, name: str) -> Lock:
        """ Get a new lock with redis.

        Use 'with lock(name):' paradigm to do the locking.

        Args:
            name (str): name of the lock.

        Returns:
            redis.lock.Lock: lock from the redis.
        """

        return self._redis.lock(name=name)


class DeploymentValidator:
    @staticmethod
    def validate_and_fill_dict(template_dict: dict, actual_dict: dict, optional_key_to_value: dict) -> None:
        """Validate incoming actual_dict with template_dict, and fill optional keys to the template.

        We use deepDiff to find missing keys in the actual_dict, see
        https://deepdiff.readthedocs.io/en/latest/diff.html#deepdiff-reference for reference.

        Args:
            template_dict (dict): template dict, we only need the layer structure of keys here, and ignore values.
            actual_dict (dict): the actual dict with values, may miss some keys.
            optional_key_to_value (dict): mapping of optional keys to values.

        Returns:
            None.
        """
        deep_diff = deepdiff.DeepDiff(template_dict, actual_dict).to_dict()

        missing_key_strs = deep_diff.get("dictionary_item_removed", [])
        for missing_key_str in missing_key_strs:
            if missing_key_str not in optional_key_to_value:
                raise Exception(f"Key '{missing_key_str}' not found.")
            else:
                DeploymentValidator._set_value(
                    original_dict=actual_dict,
                    key_list=DeploymentValidator._get_parent_to_child_key_list(deep_diff_str=missing_key_str),
                    value=optional_key_to_value[missing_key_str]
                )

    @staticmethod
    def _set_value(original_dict: dict, key_list: list, value) -> None:
        """Set the value to the original dict based on the key_list.

        Args:
            original_dict (dict): original dict that needs to be modified.
            key_list (list): the parent to child path of keys, which describes that position of the value.
            value: the value needs to be set.

        Returns:
            None.
        """
        DeploymentValidator._get_sub_structure_of_dict(original_dict, key_list[:-1])[key_list[-1]] = value

    @staticmethod
    def _get_parent_to_child_key_list(deep_diff_str: str) -> list:
        """Get parent to child key list by parsing the deep_diff_str.

        Args:
            deep_diff_str (str): a specially defined string that indicate the position of the key.
                e.g. "root['a']['b']" -> {"a": {"b": value}}.

        Returns:
            list: the parent to child path of keys.
        """

        deep_diff_str = deep_diff_str.strip("root['")
        deep_diff_str = deep_diff_str.strip("']")
        return deep_diff_str.split("']['")

    @staticmethod
    def _get_sub_structure_of_dict(original_dict: dict, key_list: list) -> dict:
        """Get sub structure of dict from original_dict and key_list using reduce.

        Args:
            original_dict (dict): original dict that needs to be modified.
            key_list (list): the parent to child path of keys, which describes that position of the value.

        Returns:
            dict: sub structure of the original_dict.
        """

        return functools.reduce(operator.getitem, key_list, original_dict)


class Subprocess:
    @staticmethod
    def run(command: str, timeout: int = None) -> None:
        """Run one-time command with subprocess.run().

        Args:
            command (str): command to be executed.
            timeout (int): timeout in seconds.

        Returns:
            str: return stdout of the command.
        """
        # TODO: Windows node
        completed_process = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout
        )
        if completed_process.returncode != 0:
            raise Exception(completed_process.stderr)
        sys.stderr.write(completed_process.stderr)

    @staticmethod
    def interactive_run(command: str) -> None:
        """Run one-time command with subprocess.popen() and write stdout output interactively.

        Args:
            command (str): command to be executed.

        Returns:
            None.
        """
        # TODO: Windows master
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
        _, stderr = process.communicate()
        if stderr:
            sys.stderr.write(stderr)


class DetailsWriter:
    @staticmethod
    def save_local_cluster_details(cluster_details: dict) -> dict:
        os.makedirs(name=f"{Paths.ABS_MARO_LOCAL}/cluster", exist_ok=True)
        with open(file=f"{Paths.ABS_MARO_LOCAL}/cluster/cluster_details.yml", mode="w") as fw:
            cluster_details = yaml.safe_dump(data=cluster_details, stream=fw)
        return cluster_details

    @staticmethod
    def save_local_master_details(master_details: dict) -> None:
        os.makedirs(name=f"{Paths.ABS_MARO_LOCAL}/cluster", exist_ok=True)
        with open(file=f"{Paths.ABS_MARO_LOCAL}/cluster/master_details.yml", mode="w") as fw:
            yaml.safe_dump(data=master_details, stream=fw)

    @staticmethod
    def save_local_node_details(node_details: dict) -> dict:
        os.makedirs(name=f"{Paths.ABS_MARO_LOCAL}/cluster", exist_ok=True)
        with open(file=f"{Paths.ABS_MARO_LOCAL}/cluster/node_details.yml", mode="w") as fw:
            node_details = yaml.safe_dump(data=node_details, stream=fw)
        return node_details


if __name__ == "__main__":
    # Load args.
    parser = argparse.ArgumentParser()
    parser.add_argument("deployment_path")
    args = parser.parse_args()

    # Load deployment and do validation.
    with open(file=os.path.expanduser(args.deployment_path), mode="r") as fr:
        join_cluster_deployment = yaml.safe_load(stream=fr)

    join_cluster_deployment = NodeJoiner.standardize_join_cluster_deployment(
        join_cluster_deployment=join_cluster_deployment
    )
    node_joiner = NodeJoiner(join_cluster_deployment=join_cluster_deployment)
    node_joiner.init_node_runtime_env()
    node_joiner.create_docker_user()
    node_joiner.setup_samba_mount()
    node_joiner.start_node_agent_service()
    node_joiner.start_node_api_server_service()
    node_joiner.copy_leave_script()
    node_joiner.load_master_public_key()

    DetailsWriter.save_local_cluster_details(cluster_details=node_joiner.cluster_details)
    DetailsWriter.save_local_master_details(master_details=node_joiner.master_details)
    DetailsWriter.save_local_node_details(node_details=node_joiner.node_details)
