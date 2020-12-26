# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import sys

from .utils.naming import generate_name_with_uuid
from .utils.subprocess import SubProcess

BUILD_IMAGE_COMMAND = """\
docker build -t {image_name} {docker_file_path}
"""

SAVE_IMAGE_COMMAND = """\
docker save {image_name} > {export_path}
"""

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_name")
    parser.add_argument("docker_file_path")
    parser.add_argument("image_name")
    args = parser.parse_args()

    # Build image
    command = BUILD_IMAGE_COMMAND.format(
        image_name=args.image_name,
        docker_file_path=f"~/.maro/clusters/{args.cluster_name}/data/{args.docker_file_path}"
    )
    return_str = SubProcess.run(command=command)
    sys.stdout.write(return_str)

    # Get image file name
    image_file_name = generate_name_with_uuid("image")

    # Save image
    command = SAVE_IMAGE_COMMAND.format(
        image_name=args.image_name,
        export_path=f"~/.maro/clusters/{args.cluster_name}/images/{image_file_name}"
    )
    _ = SubProcess.run(command=command)
    sys.stdout.write(command)
