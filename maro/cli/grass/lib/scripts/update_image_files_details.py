# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import glob
import os

from redis import Redis

from utils import load_cluster_details, get_master_details, set_master_details


def get_current_image_files_details() -> dict:
    current_images = glob.glob(os.path.expanduser(f"~/.maro/clusters/maro_grass_test/images/*"))
    image_files_details = {}

    for current_image in current_images:
        image_files_details[current_image] = {
            'modify_time': os.path.getmtime(current_image),
            'size': os.path.getsize(current_image)
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
    redis = Redis(host=master_hostname,
                  port=redis_port,
                  charset="utf-8", decode_responses=True)

    # Get details
    curr_image_files_details = get_current_image_files_details()
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
