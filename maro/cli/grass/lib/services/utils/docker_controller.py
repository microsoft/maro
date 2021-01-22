# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json

from .subprocess import Subprocess


class DockerController:
    """Controller class for docker.
    """

    @staticmethod
    def remove_container(container_name: str) -> None:
        command = f"sudo docker rm -f {container_name}"
        _ = Subprocess.run(command=command)

    @staticmethod
    def stop_container(container_name: str) -> None:
        command = f"sudo docker stop {container_name}"
        _ = Subprocess.run(command=command)

    @staticmethod
    def inspect_container(container_name: str) -> dict:
        command = f"sudo docker inspect {container_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)[0]

    @staticmethod
    def create_container_with_config(create_config: dict) -> dict:
        start_container_command = (
            "sudo docker run -it -d "
            "--cpus {cpu} "
            "--memory {memory} "
            "--name {container_name} "
            "--network host "
            "--log-driver=fluentd "
            "--log-opt tag={fluentd_tag} "
            "--log-opt fluentd-address={fluentd_address} "
            "{volumes} {environments} {labels} "
            "{image_name} {command}"
        )
        start_container_with_gpu_command = (
            "sudo docker run -it -d "
            "--cpus {cpu} "
            "--memory {memory} "
            "--gpus {gpu} "
            "--name {container_name} "
            "--network host "
            "--log-driver=fluentd "
            "--log-opt tag={fluentd_tag} "
            "--log-opt fluentd-address={fluentd_address} "
            "{volumes} {environments} {labels} "
            "{image_name} {command}"
        )

        # Format gpu params
        if "gpu" in create_config:
            start_container_command = start_container_with_gpu_command.format(gpu=create_config["gpu"])
        else:
            start_container_command = start_container_command

        # Format other params
        start_container_command = start_container_command.format(
            # User related.
            cpu=create_config["cpu"],
            memory=create_config["memory"],
            command=create_config["command"],
            image_name=create_config["image_name"],
            volumes=DockerController._build_list_params_str(params=create_config["volumes"], option="-v"),

            # System related.
            container_name=create_config["container_name"],
            fluentd_address=create_config["fluentd_address"],
            fluentd_tag=create_config["fluentd_tag"],
            environments=DockerController._build_dict_params_str(params=create_config["environments"], option="-e"),
            labels=DockerController._build_dict_params_str(params=create_config["labels"], option="-l")
        )

        # Start creating
        _ = Subprocess.run(command=start_container_command)

        # Return inspect info.
        return DockerController.inspect_container(container_name=create_config["container_name"])

    @staticmethod
    def list_container_names() -> list:
        command = "sudo docker ps -a --format \"{{.Names}}\""
        return_str = Subprocess.run(command=command)
        if return_str == "":
            return []
        return return_str.split("\n")

    @staticmethod
    def load_image(image_path: str) -> None:
        command = f"sudo docker load -q -i {image_path}"
        _ = Subprocess.run(command=command)

    # Helper functions.

    @staticmethod
    def _build_list_params_str(params: list, option: str) -> str:
        return_str = ""
        for param in params:
            return_str += f"{option} {param} "
        return return_str.strip()

    @staticmethod
    def _build_dict_params_str(params: dict, option: str) -> str:
        return_str = ""
        for k, v in params.items():
            return_str += f"{option} {k}={v} "
        return return_str.strip()
