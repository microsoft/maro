# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

from maro.cli.utils.subprocess import Subprocess


class DockerController:
    """Controller class for docker.
    """

    @staticmethod
    def save_image(image_name: str, abs_export_path: str):
        # Save image to specific folder
        os.makedirs(os.path.dirname(abs_export_path), exist_ok=True)
        command = f"docker save '{image_name}' --output '{abs_export_path}'"
        _ = Subprocess.run(command=command)
