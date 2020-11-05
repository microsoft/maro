# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool

from redis import Redis

from .utils import get_master_details, get_node_details, load_cluster_details, set_node_details

LOAD_IMAGE_COMMAND = '''\
docker load -q -i "{image_path}"
'''


def load_image(image_path: str):
    command = LOAD_IMAGE_COMMAND.format(image_path=image_path)
    completed_process = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
    )
    if completed_process.returncode != 0:
        raise Exception(completed_process.stderr)
    sys.stdout.write(completed_process.stdout)


if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_name')
    parser.add_argument('node_name')
    parser.add_argument('parallels', type=int)
    args = parser.parse_args()

    # Load details
    cluster_details = load_cluster_details(cluster_name=args.cluster_name)
    master_hostname = cluster_details['master']['hostname']
    redis_port = cluster_details['master']['redis']['port']

    # Load node details
    redis = Redis(
        host=master_hostname,
        port=redis_port,
        charset="utf-8", decode_responses=True
    )
    node_details = get_node_details(
        redis=redis,
        cluster_name=args.cluster_name,
        node_name=args.node_name
    )
    master_details = get_master_details(
        redis=redis,
        cluster_name=args.cluster_name
    )
    master_image_files_details = master_details['image_files']
    node_image_files_details = node_details['image_files']

    # Get unloaded images
    unloaded_images = []
    for image_file, image_file_details in master_image_files_details.items():
        if image_file not in node_image_files_details:
            unloaded_images.append(image_file)
        elif image_file_details['modify_time'] != node_image_files_details[image_file]['modify_time'] or \
                image_file_details['size'] != node_image_files_details[image_file]['size']:
            unloaded_images.append(image_file)
    sys.stdout.write(f"Unloaded_images: {unloaded_images}\n")
    sys.stdout.flush()

    # Parallel load
    with ThreadPool(args.parallels) as pool:
        params = [
            [os.path.expanduser(f"~/.maro/clusters/{args.cluster_name}/images/{unloaded_image}")]
            for unloaded_image in unloaded_images
        ]
        pool.starmap(
            load_image,
            params
        )

    # Save node details TODO: add lock
    node_details = get_node_details(
        redis=redis,
        cluster_name=args.cluster_name,
        node_name=args.node_name
    )
    node_details['image_files'] = master_image_files_details
    set_node_details(
        redis=redis,
        cluster_name=args.cluster_name,
        node_name=args.node_name,
        node_details=node_details
    )
