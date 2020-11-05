# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import glob
import os

from redis import Redis

from .utils import get_master_details, load_cluster_details, set_master_details


def get_current_image_files_details(cluster_name: str) -> dict:
    image_paths = glob.glob(os.path.expanduser(f"~/.maro/clusters/{cluster_name}/images/*"))
    image_files_details = {}

    for image_path in image_paths:
        file_name = os.path.basename(image_path)
        image_files_details[file_name] = {
            'modify_time': os.path.getmtime(image_path),
            'size': os.path.getsize(image_path)
        }

    return image_files_details


if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_name')
    args = parser.parse_args()

    # Load details and redis
    cluster_details = load_cluster_details(cluster_name=args.cluster_name)
    master_hostname = cluster_details['master']['hostname']
    redis_port = cluster_details['master']['redis']['port']
    redis = Redis(
        host=master_hostname,
        port=redis_port,
        charset="utf-8", decode_responses=True
    )

    # Get details
    curr_image_files_details = get_current_image_files_details(cluster_name=args.cluster_name)
    with redis.lock("lock:master"):
        master_details = get_master_details(
            redis=redis,
            cluster_name=args.cluster_name
        )
        master_details["image_files"] = curr_image_files_details
        set_master_details(
            redis=redis,
            cluster_name=args.cluster_name,
            master_details=master_details
        )
