# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import subprocess
import sys

from .utils import generate_name_with_uuid

build_image_command = """\
docker build -t {image_name} {docker_file_path}
"""

save_image_command = """\
docker save {image_name} > {export_path}
"""

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_name')
    parser.add_argument('docker_file_path')
    parser.add_argument('image_name')
    args = parser.parse_args()

    # Build image
    command = build_image_command.format(
        image_name=args.image_name,
        docker_file_path=f"~/.maro/clusters/{args.cluster_name}/data/{args.docker_file_path}"
    )
    completed_process = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
    )
    if completed_process.returncode != 0:
        raise Exception(completed_process.stderr)
    sys.stdout.write(command)

    # Get image file name
    image_file_name = generate_name_with_uuid("image")

    # Save image
    command = save_image_command.format(
        image_name=args.image_name,
        export_path=f"~/.maro/clusters/{args.cluster_name}/images/{image_file_name}"
    )
    completed_process = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
    )
    if completed_process.returncode != 0:
        raise Exception(completed_process.stderr)
    sys.stdout.write(command)
